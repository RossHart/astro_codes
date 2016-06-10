import glob, os

for file in glob.glob('*.pdf'):
    os.system('pdfcrop {} {}'.format(file,file))