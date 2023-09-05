#!/usr/bin env python
import sys
import pygame
import numpy as np

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

SCREEN_SIZE = (640, 640)
FPS = 60


class Sim():
    def __init__(self, sim_dt, fps, world_dims) -> None:
        self.objects = []
        self.sim_dt = sim_dt
        self.fps = fps
        self.ground_y = None
        self.world_dims = world_dims
        self.world_lows = self.world_dims[:,0]
        self.world_highs = self.world_dims[:,1]

        self.res  = (self.world_highs - self.world_lows)/(np.array(SCREEN_SIZE)*1.)
        
        self.orig_pix = np.floor(0 - self.world_lows/self.res)
        print(self.res, self.orig_pix, self.world_lows, self.world_highs)

        # create a surface on screen that has the size screen_size
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        #get clock object to regulate frame-rate
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Dynamics Sim')


    def add_ground_plane(self, ground_y):
        self.ground_y = ground_y

    def add_object(self, object):
        self.objects.append(object)
    
    def enforce_bounds(self, object):
        #enforce bounds
        if self.ground_y is not None:
            if object.position[1] <= self.ground_y + object.radius:
                object.position[1] = self.ground_y + object.radius
                object.acceleration[-1] = 0.0
                object.velocity[-1] = 0.0
            # ball.position[1] = ground_y
        if (object.position[0] <= self.world_lows[0] + object.radius) or (object.position[0] >= self.world_highs[0] - object.radius):
            object.acceleration[-2] = 0.0
            object.velocity[-2] = -object.velocity[0]
        
    def step(self, f_ext):
        dt_ms = self.clock.tick(self.fps)
        
        self.screen.fill((WHITE))
        if self.ground_y is not None:
            #draw ground plane
            ground_begin = np.array([self.world_lows[0], self.ground_y])
            ground_end = np.array([self.world_highs[1], self.ground_y])
            pygame.draw.line(
                self.screen, BLACK, 
                self.world_to_pixel(ground_begin), 
                self.world_to_pixel(ground_end))

        if len(self.objects) > 0:
            for obj in self.objects:
                obj.apply_input(f_ext)
                self.enforce_bounds(obj)
                obj.apply_forces()
                obj.step(self.sim_dt)
                pixel_radius = (SCREEN_SIZE[0] * obj.radius) / (self.world_highs[0] - self.world_lows[0]) 

                pygame.draw.circle(self.screen, obj.color, self.world_to_pixel(obj.pos), 
                                   pixel_radius, obj.width)

                # object.display(self.screen)
        pygame.display.flip()


    def world_to_pixel(self, world_coords):
        pix_x = int(self.orig_pix[0] + np.floor(world_coords[0]/self.res[0]))
        pix_y = int(SCREEN_SIZE[1] - (self.orig_pix[1] + np.floor(world_coords[1]/self.res[1])))
        return np.array((pix_x, pix_y))

    def pixel_to_world(self, pixel_coords):
        world_x = (pixel_coords[0] - self.orig_pix[0]) * self.res[0]
        world_y = (pixel_coords[1] - self.orig_pix[1])* self.res[1]
        return np.array([world_x, world_y])

class Circle():
    def __init__(
            self, 
            pos, 
            vel=np.zeros(2), 
            acc=np.zeros(2),
            radius=0.5, 
            mass=0.1, 
            color=WHITE, width=10):
        
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.input = None
        self.radius = radius
        self.mass = mass
        self.color = color
        self.width = width

    def step_vel(self, velocity, dt):
        self.vel = velocity
        self.pos += velocity * dt
    
    def step_acc(self, acc, dt):
        self.acc = acc
        self.vel += acc * dt
        self.pos += self.vel * dt
    
    def step(self, dt):
        self.vel += self.acc * dt
        self.pos += self.vel[2:4] * dt

    def apply_input(self, f_ext):
        self.input = f_ext
    
    def apply_forces(self):
        linear_acc = (2.0 * self.input) / 3.0
        angular_acc = self.radius * linear_acc
        self.acc = np.concatenate([angular_acc, linear_acc])
        
    @property
    def position(self):
        return self.pos

    @property 
    def velocity(self):
        return self.vel
    
    @property 
    def acceleration(self):
        return self.acc

    
def main():
    pygame.init()

    ground_y = 0.0
    gravity_acc = np.array([0., -9.81]) #m/s^2
    world_dims = np.array([[-1., 1.],[-1., 1.]])
    sim = Sim(0.01, FPS, world_dims)
    sim.add_ground_plane(ground_y)

    #create ball object
    ball_init_pos = np.array([-0.1, 0.5]) #m
    ball_init_vel = np.array([0.0, 0.0, 0.0, 0.0]) #m/s
    ball_init_acc = np.array([0.0, 0.0, 0.0, 0.0]) #m/s^2
    ball = Circle(
        pos=ball_init_pos,
        vel=ball_init_vel,
        acc=ball_init_acc, 
        radius=0.1, mass=0.1, 
        color=BLUE)
    
    sim.add_object(ball)
    obj_mass = ball.mass
    
    f_ext = np.array([0.0, 0.0]) + obj_mass * gravity_acc
    running = True 
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    f_ext = np.array([-1.0, 0.0]) +  obj_mass * gravity_acc
                if event.key == pygame.K_RIGHT:
                    f_ext = np.array([1.0, 0.0]) +  obj_mass*gravity_acc
        
        # if ball.position[1] > ground_y + ball.radius:
        #     f_ext = obj_mass * gravity_acc
        # else:
        #     f_ext = np.array([1.0, 0.0])

        sim.step(f_ext)
        f_ext = np.array([0.0, 0.0]) + obj_mass * gravity_acc


    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()