import os
import glob
import librosa
import scipy.io.wavfile as wavfile
import multiprocessing


def process_file(f, root_dst, fs_trg=22050):
    dirname = os.path.dirname(f)
    ff = f.replace(dirname, root_dst)
    if os.path.isfile(ff):
        return True
    if os.path.getsize(f) < 100:
        return False
    x, fs = librosa.core.load(f, sr=None)
    if x.shape[0] == 0:
        return False
    if fs != fs_trg:
        x = librosa.core.resample(x, fs, fs_trg)
    else:
        pass
    wavfile.write(ff, rate=fs_trg, data=x)
    return True


def resample_mp(rootp, root_dstp, fs_trg):
    fnames = glob.glob(rootp + '/*.wav', recursive=True)
    print("found ", len(fnames))
    if not os.path.isdir(root_dstp):
        os.mkdir(root_dstp)
    p = multiprocessing.Pool()
    for i, f in enumerate(fnames):
        p.apply_async(process_file, [f, root_dstp, fs_trg])
    p.close()
    p.join()


#if __name__ == '__main__':
 #   fs_trg = 22050
 #  root = r'data/UrbanSound8K/audio'
 #  root_dst = root + '_22_5'
 #  resample_mp(root, root_dst, fs_trg)
 #  print("DONE")
    
if __name__ == '__main__':
    fs_trg = 22050
    folds=["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8","fold9","fold10"]
    root = r'data/UrbanSound8K/audio'
    root_dst = root + '_22_5'
    os.mkdir(root_dst)
    for i in folds:
        rootp=root+'/'+i
        root_dstp=root_dst+'/'+i
        resample_mp(rootp, root_dstp, fs_trg)
    print("DONE")