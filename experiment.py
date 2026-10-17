"""Generate all predeclared images, run real local Tesseract, and calculate CER.
No external OCR API is used. Run on Ubuntu 24.04 with the pinned packages in README.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,os,platform,re,shutil,subprocess
import PIL
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())
TEXTS=json.loads((ROOT/'texts.json').read_text())
for name in ('data','images','raw','normalized','references','fonts','models'):
    (ROOT/name).mkdir(exist_ok=True)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalize(text):return ' '.join(text.split())

def align(ref,hyp):
    # Unit-cost Levenshtein DP; ties prefer substitution, deletion, insertion.
    n,m=len(ref),len(hyp)
    dp=[list(range(m+1))]+[[i]+[0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(ref[i-1]!=hyp[j-1]))
    edits=[];pairs=[];i,j=n,m
    while i or j:
        if i and j and ref[i-1]==hyp[j-1] and dp[i][j]==dp[i-1][j-1]:
            pairs.append((ref[i-1],hyp[j-1]));i-=1;j-=1
        elif i and j and dp[i][j]==dp[i-1][j-1]+1:
            edits.append({'type':'S','reference_index':i-1,'ocr_index':j-1,'reference':ref[i-1],'ocr':hyp[j-1]})
            pairs.append((ref[i-1],hyp[j-1]));i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:
            edits.append({'type':'D','reference_index':i-1,'ocr_index':j,'reference':ref[i-1],'ocr':''})
            pairs.append((ref[i-1],''));i-=1
        else:
            edits.append({'type':'I','reference_index':i,'ocr_index':j-1,'reference':'','ocr':hyp[j-1]})
            pairs.append(('',hyp[j-1]));j-=1
    pairs.reverse();edits.reverse()
    assert ''.join(x[0] for x in pairs)==ref and ''.join(x[1] for x in pairs)==hyp
    assert len(edits)==dp[n][m]
    return dp[n][m],edits

def render(text,font):
    im=Image.new('L',tuple(PLAN['original_size']),255);d=ImageDraw.Draw(im)
    margin=PLAN['margin_pixels'];maxwidth=im.width-2*margin
    lines=[];line=''
    for word in text.split():
        candidate=(line+' '+word).strip()
        if line and d.textlength(candidate,font=font)>maxwidth:lines.append(line);line=word
        else:line=candidate
    lines.append(line)
    y=margin
    for line in lines:
        assert y+PLAN['line_height_pixels']<im.height-margin
        d.text((margin,y),line,font=font,fill=0);y+=PLAN['line_height_pixels']
    return im,lines

def main():
    exe=shutil.which('tesseract')
    if not exe:raise RuntimeError('Tesseract command line program is required')
    version=subprocess.run([exe,'--version'],capture_output=True,text=True,check=True).stdout
    assert version.splitlines()[0]=='tesseract 5.3.4',version
    font=ROOT/'fonts/DejaVuSans.ttf'
    if not font.exists():shutil.copyfile('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',font)
    license_path=ROOT/'fonts/LICENSE-DejaVu.txt'
    if not license_path.exists():shutil.copyfile('/usr/share/doc/fonts-dejavu-core/copyright',license_path)
    model=ROOT/'models/eng.traineddata'
    if not model.exists():shutil.copyfile('/usr/share/tesseract-ocr/5/tessdata/eng.traineddata',model)
    model_license=ROOT/'models/LICENSE-tessdata.txt'
    if not model_license.exists():shutil.copyfile('/usr/share/doc/tesseract-ocr-eng/copyright',model_license)
    f=ImageFont.truetype(str(font),PLAN['font_size_pixels'])
    # Record environment and immutable design BEFORE any main image reaches OCR.
    environment={'tesseract_version':version,'python':platform.python_version(),'Pillow':PIL.__version__,
      'platform':platform.platform(),'language_model_sha256':digest(model),'font_sha256':digest(font),
      'plan_sha256':digest(ROOT/'plan.json'),'texts_sha256':digest(ROOT/'texts.json'),
      'packages':subprocess.run(['dpkg-query','-W','tesseract-ocr','tesseract-ocr-eng','fonts-dejavu-core'],capture_output=True,text=True,check=True).stdout}
    (ROOT/'data/environment.json').write_text(json.dumps(environment,indent=2))
    rows=[];all_edits={};manifest=[]
    for sample,text in TEXTS.items():
        assert 80<=len(text.split())<=120,(sample,len(text.split()))
        (ROOT/f'references/{sample}.txt').write_text(text+'\n',encoding='utf-8')
        ref=normalize(text);(ROOT/f'normalized/{sample}-reference.txt').write_text(ref,encoding='utf-8')
        original,wrapped=render(text,f)
        variants={'original':original,
          'low-resolution':original.resize(tuple(PLAN['low_resolution_size']),Image.Resampling.LANCZOS),
          'skewed':original.rotate(PLAN['skew_degrees_counterclockwise'],resample=Image.Resampling.BICUBIC,expand=True,fillcolor=255)}
        for condition,im in variants.items():
            key=f'{sample}-{condition}';relative=f'images/{key}.png';path=ROOT/relative
            im.save(path,dpi=(300,300))
            command=[exe,relative,f'raw/{key}','--tessdata-dir','models','-l','eng','--oem','1','--psm','6','--dpi','300']
            execution=subprocess.run(command,cwd=ROOT,capture_output=True,check=True,env={**os.environ,'OMP_THREAD_LIMIT':'1'})
            (ROOT/f'raw/{key}.stderr.txt').write_bytes(execution.stderr)
            raw=(ROOT/f'raw/{key}.txt').read_text(encoding='utf-8')
            hyp=normalize(raw);(ROOT/f'normalized/{key}.txt').write_text(hyp,encoding='utf-8')
            distance,edits=align(ref,hyp);counts=Counter(e['type'] for e in edits)
            example=None
            if edits:
                e=edits[0];a,b=e['reference_index'],e['ocr_index']
                example={**e,'reference_context':ref[max(0,a-28):min(len(ref),a+35)],'ocr_context':hyp[max(0,b-28):min(len(hyp),b+35)]}
            row={'sample':sample,'condition':condition,'width':im.width,'height':im.height,
              'angle_degrees':3 if condition=='skewed' else 0,'reference_characters':len(ref),
              'ocr_characters':len(hyp),'substitutions':counts['S'],'deletions':counts['D'],'insertions':counts['I'],
              'character_errors':distance,'cer':distance/len(ref),'cer_percent':100*distance/len(ref),
              'image':relative,'raw_text':f'raw/{key}.txt','normalized_text':f'normalized/{key}.txt',
              'reference_text':f'references/{sample}.txt','image_sha256':digest(path),'example':example,
              'command':command[1:]}
            rows.append(row);all_edits[key]=edits
    assert len(rows)==9
    # Self-contained results and a manifest of all evidence files, without machine-specific paths.
    for folder in ('images','raw','normalized','references','fonts','models'):
        for path in sorted((ROOT/folder).glob('*')):
            manifest.append({'file':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)})
    result={'plan':PLAN,'environment':environment,'rows':rows}
    (ROOT/'data/results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'data/edit-operations.json').write_text(json.dumps(all_edits,indent=2),encoding='utf-8')
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (ROOT/'data/results.csv').open('w',newline='',encoding='utf-8') as file:
        columns=['sample','condition','width','height','angle_degrees','reference_characters','ocr_characters','substitutions','deletions','insertions','character_errors','cer_percent','image','raw_text','normalized_text']
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'versions':environment,'results':[{k:r[k] for k in ('sample','condition','character_errors','cer_percent')} for r in rows]},indent=2))

if __name__=='__main__':main()
