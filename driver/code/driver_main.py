from deep_pi_car import DeepPiCar
import logging
import sys

def main():
    # print system info
    logging.info('Starting DeepPiCar, system info: ' + sys.version)
    
    with DeepPiCar() as car:
        car.drive(10)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler("dpc.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    main()
