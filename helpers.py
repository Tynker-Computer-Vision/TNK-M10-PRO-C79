import math 

def getSensorX(angle, sensorAngle):
    return math.cos(math.radians(angle+sensorAngle))

def getSensorY(angle, sensorAngle):
    return math.sin(math.radians(angle+sensorAngle))