import IMU

IMU.detectIMU()
IMU.initIMU()
while True:
    horizontal = IMU.readACCx()
    vertical = IMU.readACCz() - 4000
    if horizontal > vertical:
        print("front/back")
    else:
        print("up/down")
