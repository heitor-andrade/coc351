import pygame
from pygame.locals import *
import sys
import math
import numpy as np


def bissecao_pts(f, a, b, xtol=1e-8, ytol=1e-8, maxiter=100, trace=False):
    """
    Encontra uma raiz de  f  pelo algoritmo da bissecao, retornando
    todos os pontos pelos quais o algoritmo passa.
    """
    m = (a+b)/2
    m_list = [m]
    _xtol = abs(b-a)
    _ytol = abs(f(m))
    i = 0
    while (_xtol >= xtol and _ytol >= ytol and i <= maxiter):
        # verifica se os sinais tão invertidos com o ponto médio
        # e escolhe a bisseção correta
        if (f(a) * f(m) < 0):
            b = m
        else:
            a = m

        # calcula o ponto médio do novo intervalo
        m = (a+b)/2
        m_list.append(m)

        # calcula tolerancia do intervalo atual
        _xtol = abs(b-a)
        _ytol = abs(f(m))
        i += 1

    return m_list if trace else m_list[-1]


WIDTH, HEIGHT = 840, 456
FPS = 60
DT = 1/FPS

fps = pygame.time.Clock()
pygame.init()
display = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sinuca")

# Setting up color objects
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

RADIUS_BALL = 15.6 # milimetros
WEIGHT_BALL = 128.5  # gramas
RADIUS_WHITE_BALL = 16.8
BETA = DT


class Ball():
    def __init__(self, color, center, radius, mass=128.5):
        self.color = color
        self.center = center
        self.radius = radius
        self.velocity = [0, 0]
        self.fat = 1
        self.metodo = 0

    def updatePosition(self):
        for i, vel in enumerate(self.velocity):
            if abs(vel) < 12 * self.fat/FPS:
                self.velocity[i] = 0
            else:
                # sign = vel * (-1)
                # self.velocity[i] = vel + self.fat/FPS*sign
                sign = vel * (-1)/abs(vel)
                self.velocity[i] = vel + (0.1 + abs(vel*BETA))*sign


        if self.metodo == 0:
            self.move_euler()
        elif self.metodo == 1:
            self.move_heun()
        elif self.metodo == 2:
            self.move_RK4()

    def nextVelocity(self, vel):
        sign = vel * (-1)/abs(vel)
        return vel + (0.1 + abs(vel*BETA))*sign

    def move_euler(self):
        self.center[0] += self.velocity[0]*DT
        self.center[1] += self.velocity[1]*DT

    def move_heun(self):
         
        velocity_segunda = [0,0]
        velocity_segunda[0] = self.nextVelocity(self.velocity[0])
        velocity_segunda[1] = self.nextVelocity(self.velocity[1])

        self.center[0] += (self.velocity[0] + velocity_segunda[1])*DT/2
        self.center[1] += (self.velocity[1] + velocity_segunda[1])*DT/2
    
    def move_RK4(self):
        k1 = [self.velocity[0], self.velocity[1]]
        k2 = [self.nextVelocity(k1[0]), self.nextVelocity(k1[1])]
        k3 = [self.nextVelocity(k2[0]), self.nextVelocity(k2[1])]
        k4 = [self.nextVelocity(k3[0]), self.nextVelocity(k3[1])]

        self.center[0] += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0] )*DT/6
        self.center[1] += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1] )*DT/6

    def move(self, x, y):
        self.velocity = [x, y]


    def draw(self, display):
        pygame.draw.circle(display, self.color, self.center, self.radius)


def make16Balls():
    N_BALLS = 16
    x_ball, x_ball_init = WIDTH*2/3, WIDTH*2/3
    y_ball, y_ball_init = HEIGHT/2, HEIGHT/2
    qtde_total, qtde_linha = 0, 0
    balls_list = []
    for i in range(N_BALLS-1):
        balls_list.append(Ball(BLUE, [x_ball, y_ball], RADIUS_BALL))
        if(i == qtde_total + qtde_linha):
            qtde_linha += 1
            qtde_total += qtde_linha
            x_ball_init = x_ball_init + RADIUS_BALL*math.sqrt(3)
            y_ball_init = y_ball_init - RADIUS_BALL
            x_ball, y_ball = x_ball_init, y_ball_init
        else:
            y_ball += RADIUS_BALL*2

    balls_list.append(
        Ball((255, 255, 0), [RADIUS_BALL, HEIGHT/2], RADIUS_BALL))

    return balls_list


def handleCollisionEdges(ball):
    if ball.center[0] - 15 <= 0:
        ball.center[0] = 15.01  # Desloca a bola para fora da parede
        ball.velocity[0] *= -1

    elif ball.center[0] + 15 >= WIDTH:
        ball.center[0] = WIDTH - 15.01
        ball.velocity[0] *= -1

    elif ball.center[1] - 15 <= 0:
        ball.center[1] = 15.01
        ball.velocity[1] *= -1

    elif ball.center[1] + 15 >= HEIGHT:
        ball.center[1] = HEIGHT - 15.01
        ball.velocity[1] *= -1

    else:
        return False

    return True

atacando = False
first_mouse_pos = (None, None)
fpc = False


def handleTacada(mouse_pos, white_ball):
    kp = 3
    global fpc
    global first_mouse_pos
    global atacando
    # se o mouse estiver segurado, vai pegando a posição relativa
    if(atacando):
        if(fpc):
            if not pygame.mouse.get_pressed()[0]:
                x = mouse_pos[0] - first_mouse_pos[0]
                y = mouse_pos[1] - first_mouse_pos[1]
                print("velocidades: x, y", -x*kp, -y*kp)
                white_ball.move(-x*kp, -y*kp)
                atacando = False
                fpc = False
        else:
            fpc = True
            first_mouse_pos = mouse_pos


def drawBalls(ball_list, display):
    for i, ball in enumerate(ball_list):
        ball.updatePosition()
        ball.draw(display)
        handleCollisionEdges(ball)

        for j in range(len(ball_list)):
            if(i != j):
                handleCollisionsBalls(ball, balls_list[j])


def make_function(ball1, ball2):
    # retorna a função que possui como raiz o tempo de colisão de ball1 com ball2
    x1, y1 = ball1.center
    x2, y2 = ball2.center
    v1x, v1y = ball1.velocity
    v2x, v2y = ball2.velocity

    def f(t):
        pos_x1 = x1 + v1x*t
        pos_x2 = x2 + v2x*t
        dif_x = pos_x1 - pos_x2

        pos_y1 = y1 + v1y*t
        pos_y2 = y2 + v2y*t
        dif_y = pos_y1 - pos_y2
        return math.sqrt(dif_x**2 + dif_y**2) - (ball1.radius + ball2.radius + 1e-2)
    return f


def changePosition(ball1, ball2, t):
    # pega o tempo e altera os centros das bolas para o ponto de colisão
    ball1.center[0] = ball1.center[0] - ball1.velocity[0]*(t + t*1e-2)
    ball1.center[1] = ball1.center[1] - ball1.velocity[1]*(t + t*1e-2)
    ball2.center[0] = ball2.center[0] - ball2.velocity[0]*(t + t*1e-2)
    ball2.center[1] = ball2.center[1] - ball2.velocity[1]*(t + t*1e-2)


def handleCollisionsBalls(ball1, ball2):
    x_2 = (ball1.center[0] - ball2.center[0])**2
    y_2 = (ball1.center[1] - ball2.center[1])**2
    dist = math.sqrt(x_2+y_2)

    v1, c1 = np.asarray(ball1.velocity), np.asarray(ball1.center)
    v2, c2 = np.asarray(ball2.velocity), np.asarray(ball2.center)

    if dist <= (ball1.radius + ball2.radius):
        f = make_function(ball1, ball2)
        t = bissecao_pts(f, 0, -1.8*DT)
        changePosition(ball1, ball2, t)
        ball1.velocity = v1 - (np.dot(v1-v2, c1-c2) /
                               (np.linalg.norm(c1-c2)**2))*(c1-c2)
        ball2.velocity = v2 - (np.dot(v2-v1, c2-c1) /
                               (np.linalg.norm(c2-c1)**2))*(c2-c1)


white_ball = Ball(WHITE, [WIDTH/3, HEIGHT/2], RADIUS_WHITE_BALL)
balls_list = make16Balls()
balls_list.append(white_ball)

def restart():
    white_ball.center[0], white_ball.center[1] = WIDTH/10, HEIGHT/2
    test_ball.center[0], test_ball.center[1] = WIDTH/2, HEIGHT/2
    
    white_ball.velocity[0], white_ball.velocity[1] = 0,0
    test_ball.velocity[0], test_ball.velocity[1] = 0,0


def isStoped():
    soma = 0
    for ball in balls_list:
        soma += sum(ball.velocity)
    return soma == 0

def tacada_x(vx):
    white_ball.move(vx, 0)


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if pygame.mouse.get_pressed()[0]:
            atacando = True
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.key.key_code("r"):
                restart()
            if event.key == pygame.key.key_code("k"):
                tacada_x(600)
            if event.key == pygame.key.key_code("c"):
                j += 1
                if j> 2:
                    j = 0


    display.fill(BLACK)
    handleTacada(pygame.mouse.get_pos(), white_ball)
    drawBalls(balls_list, display)
    pygame.display.update()
    fps.tick(60)