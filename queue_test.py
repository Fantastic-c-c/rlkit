import multiprocessing as mp

class Test:
    def __init__(self):
        self.queue = mp.Queue()

    def worker(self, q):
        q.put("ABC")

    def main(self):

        # p = mp.Process(target=self.worker, args=())
        p = mp.Process(target=self.worker, args=(self.queue, ))
        p.start()
        while self.queue.empty():
            pass
        print(self.queue.get())
        p.join()

if __name__ == '__main__':
    mp.freeze_support()
    t = Test()
    t.main()
