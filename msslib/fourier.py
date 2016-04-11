import numpy as np

def local_fft(img, point, window):
    half = window / 2
    x, y = point
    x = x - half
    y = y - half
    return np.abs(np.fft.fft2(img[x:x+window+1, y:y+window+1]))


def avg_fft(img, point, window, step):
    """ Average FFT of windows around a point"""
    x,y = point
    count = 0
    acc = np.zeros((window,window), dtype='float64')
    for i in range(x-window,x+1, step):
        for j in range(y-window,y+1, step):
            sl = np.s_[i:i+window,j:j+window]
            f = np.fft.fft2(img[sl])
            acc = acc + np.abs(f)
            count += 1
    return acc / count
