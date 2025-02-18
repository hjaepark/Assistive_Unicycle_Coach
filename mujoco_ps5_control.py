import pygame

pygame.init()
pygame.joystick.init()

joystick_count = pygame.joystick.get_count()
if joystick_count < 1:
    raise RuntimeError("Joystick not found")

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Joystick Name:", joystick.get_name())
print("Number of Axes:", joystick.get_numaxes())
print("Number of Buttons:", joystick.get_numbuttons())

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.JOYAXISMOTION:
            axis_idx = event.axis
            axis_val = event.value  # -1.0 ~ 1.0
            
            print(f"Axis {axis_idx} moved to {axis_val:.3f}")
            

    
pygame.quit()
