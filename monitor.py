import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import numpy as np

class ConvertEvent(FileSystemEventHandler):
    def on_modified(self, event):
        
        time.sleep(60)

        test = np.loadtxt('N_proection_alphabet_data/outputs/test.txt')
        output = np.loadtxt('N_proection_alphabet_data/outputs/output.txt')

        test = np.reshape(test, (32, 32))
        output = np.reshape(output, (32, 32))

        test *= 255
        output *= 255

        test = Image.fromarray(test.astype('uint8')).convert('L')
        output = Image.fromarray(output.astype('uint8')).convert('L')

        test.save('N_proection_alphabet_data/outputs/test.png')
        output.save('N_proection_alphabet_data/outputs/output.png')

if __name__ == "__main__":
    path = 'N_proection_alphabet_data/outputs/'
    event_handler = ConvertEvent()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
