import pygame,math
import neat

import pickle

pygame.init()
screen = pygame.display.set_mode((800,600))

pygame.display.set_caption("Car racing")
background_image = pygame.image.load("track.png").convert()
player_image = pygame.image.load("car.png").convert_alpha()

player=pygame.Rect(60,300,20,20)

WHITE=(255,255,255)
xvel=2
yvel=3
angle=0
change=0

# Creating a variable 'distance' and assigning value '5' to it
distance=2
# Creating a variable 'forward' and assigning 'False' to it
forward=False

font = pygame.font.Font('freesansbold.ttf', 12)

# Defining a function 'newxy()' to calculate new x,y coordinates
# New x,y coordinates are based on x,y coordinates, angle, distance 
def newxy(x,y,distance,angle):
  angle=math.radians(angle+90)
  xnew=x+(distance*math.cos(angle))
  # Calculating the new y-coordinate value 'ynew'
  ynew=y-(distance*math.sin(angle))
  # Returning 'xnew','ynew'
  return xnew,ynew

def checkOutOfBounds(car):
  x = car.x
  y = car.y
  width = car.width
  height = car.height

  if(checkPixel(x,y) or checkPixel(x+width, y) or checkPixel(x, y+height) or checkPixel(x+width, y+height)):
      return True
  
def checkPixel(x, y):
    global screen
    try:
        color = screen.get_at((x, y))
    except:
        return 1
    if(color == (163,171,160,255)):
        return 0
    return 1

def getCorners(car, angle):
    global screen
    margin = 55
    delta = 5
    x = car.x + car.w/2
    y = car.y + car.h/2
    
    sensorAngles = [-10,-30,-50,-70,-90,-110,-130,-150,-170]
    sensorData= []
    for sensorAngle in sensorAngles:
        newX = int(x -(margin * math.cos(math.radians(angle+sensorAngle))))
        newY = int(y +(margin * math.sin(math.radians(angle+sensorAngle))))
        
        
        sensorData.append(checkPixel(newX, newY))
         
        pygame.draw.rect(screen,(0, 255,0), [newX, newY, 5, 5])
        margin = margin + delta
        if margin > 70:
            delta = -delta
    
    print(sensorData)
    return sensorData[0], sensorData[1], sensorData[2], sensorData[3], sensorData[4], sensorData[5], sensorData[6], sensorData[7] 
    
gen=0
angle =0

def save(winner):
    file_name = 'std1.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(winner, file)
        print(f'Object successfully saved to "{file_name}"')
  
def eval_fitness(generation, config):
    global angle, gen, forward, change
    gen = gen+1
    genomeCount = 1
    # Printing the generation count 
    print("Generation:",gen, "Total", len(generation) )
    
    for gid, genome in generation:
        
        genome.fitness = 0 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        infoText = font.render('Generation('+str(len(generation))+") :"+ str(gen)+  ' genomecount:' +str(genomeCount)+ str(genome) , True, (255,255,0))

        print(genome)
        while True:
          screen.blit(background_image,[0,0])
          screen.blit(infoText, (220, 20))
          fitnessText = font.render('fitness Score:'+ str(genome.fitness) , True, (255,255,0))
          screen.blit(fitnessText, (420, 40))
          for event in pygame.event.get():
            if event.type == pygame.QUIT:
              pygame.quit()
              
            if event.type == pygame.KEYDOWN:
               if event.key == pygame.K_LEFT:
                  change = 5
               if event.key ==pygame.K_RIGHT:
                change = -5 
               # Checking if UP arrow key is pressed and make 'forward' to True
               if event.key == pygame.K_UP:
                forward = True
                
            if event.type == pygame.KEYUP:
              if event.key ==pygame.K_LEFT or event.key == pygame.K_RIGHT:
                  change = 0
              # Checking if UP arrow key is released and make 'forward' to False
              if event.key == pygame.K_UP:
                forward = False 
            
          # Checking if 'forward' is 'True' 
          if forward:
              # Finding new x,y coordinates by calling the 'newxy()' function
              player.x,player.y=newxy(player.x, player.y, 3, angle)  
                          
          if(checkOutOfBounds(player)):
              player.x = 60
              player.y = 300
              angle =0
              genomeCount = genomeCount +1
              break
          
          angle = angle + change
          
          newimage=pygame.transform.rotate(player_image,angle) 
          pygame.draw.rect(screen,(0, 255, 0), player)
          screen.blit(newimage ,player)
            
          # Controlling the game using neural net
          forward = True
          change = 0
          
          corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8 = getCorners(player, angle)
          
          output = net.activate((angle, corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8))
          
          
          inputText = font.render('All Sensors:'+ str(corner1)+ str(corner2)+ str(corner3)+ str(corner4)+ str(corner5)+ str(corner6)+ str(corner7)+ str(corner8) , True, (255,255,0))
          screen.blit(inputText, (420, 60))
          
          output1Text = font.render('Output1:'+ str(output[0]), True, (255,255,0))
          screen.blit(output1Text, (420, 80))
          output2Text = font.render('Output2:'+ str(output[1]), True, (255,255,0))
          screen.blit(output2Text, (420, 100))
          if output[0] > 0.65:
              change = 3
          if output[1] > 0.65:
              change = -3
              
          genome.fitness += 0.2
          
          if(genome.fitness>1000):
              save(genome)
              break
          
          pygame.display.update()
          pygame.time.Clock().tick(30)
  
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward.txt')  
p = neat.Population(config)
winner = p.run(eval_fitness,10) 

save(winner)