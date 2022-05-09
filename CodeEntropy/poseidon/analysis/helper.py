import psutil
import os

def memoryInfo(verbosePrint):
    '''
    '''
    process = psutil.Process(os.getpid())
    bytes_info = float(process.memory_info().rss)
    gb = bytes_info * 1e-9
    print('memory use:', round(gb, 3), 'GB')  # in gbytes




def weightingPopulation(weighting):
    '''
    Save a list of weightings per frame from biased simulations in the
    EEclass
    '''
    dataFile = weighting[0]
    try:
        frame_chunks = weighting[1]
    except IndexError:
        frame_chunks = 1
    weighting_list = []
    with open(dataFile) as data:
        for line in data:
            weighting_list.append(line)
    data.close()

    weighting_info = [weighting_list, frame_chunks]

    return weighting_info
