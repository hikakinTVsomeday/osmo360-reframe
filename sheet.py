import subprocess, sys, os, json
from PIL import Image, ImageDraw
files = sys.argv[2:]
out = sys.argv[1]
rows = []
for f in files:
    d = float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',f]).decode().strip())
    ims = []
    for p in (0.1,0.5,0.9):
        tmp = 'tmp_%d.jpg' % int(p*100)
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(d*p),'-i',f,'-map','0:v:0','-frames:v','1','-vf','scale=480:-2',tmp],check=True)
        ims.append(Image.open(tmp).convert('RGB'))
    h = max(i.height for i in ims)
    row = Image.new('RGB',(480*3,h+24),'black')
    for k,i in enumerate(ims): row.paste(i,(480*k,24))
    ImageDraw.Draw(row).text((5,5),os.path.basename(f)+'  %.0fs'%d,fill='yellow')
    rows.append(row)
W = max(r.width for r in rows); H = sum(r.height for r in rows)
sheet = Image.new('RGB',(W,H),'black'); y=0
for r in rows: sheet.paste(r,(0,y)); y+=r.height
sheet.save(out, quality=80)
print(sheet.size)
