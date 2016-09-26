import glob, os

def crop_all():
    for filename in glob.glob('*.pdf'):
        os.system('pdfcrop {} {}'.format(filename,filename))
    return None

def crop(filename):
    os.system('pdfcrop {} {}'.format(filename,filename))
    return None