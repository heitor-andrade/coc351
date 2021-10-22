import matplotlib.pyplot as plt

FPS = 60
DT = 1/60
BETA = DT

class Ball():
    def __init__(self, center = 0):
        self.center = center
        self.velocity = 0
        self.fat = 1
        self.metodo = 0


    def updatePosition(self):
        vel = self.velocity
        if abs(vel) < 2:
            self.velocity = 0
        else:
            self.velocity = self.nextVelocity(self.velocity)

        if self.metodo == 0:
            self.move_euler()
        elif self.metodo == 1:
            self.move_heun()
        elif self.metodo == 2:
            self.move_RK4()

    def nextVelocity(self, vel):
        if vel != 0:
            sign = vel * (-1)/abs(vel)
            vel = vel + (0.01 + abs(vel*BETA))*sign
            return vel
        return 0

                
    def move_euler(self):
        self.center += self.velocity*DT

    def move_heun(self):
        velocity_segunda = self.nextVelocity(self.velocity)

        self.center += (self.velocity + velocity_segunda)*DT/2

    def move_RK4(self):
        k1 = self.velocity
        k2 = self.nextVelocity(k1)
        k3 = self.nextVelocity(k2)
        k4 = self.nextVelocity(k3)

        self.center += (k1 + 2*k2 + 2*k3 + k4 )*DT/6

    def move(self, x):
        self.velocity = x

    def isStopped(self):
        return self.velocity == 0

ball = Ball()

VELOCIDADE_INICIAL = 300
PASSO = 50

x_pos = []
vel = []

def tacada(ball, vel, metodo):
    ball.metodo = metodo
    ball.move(vel)
    
    while not ball.isStopped():
        ball.updatePosition()

    x = ball.center
    ball.center = 0    
    return x


euler_dist = []
heun_dist = []
rk4_dist = []
velocidades = []

for i in range(20):
    euler_dist.append(tacada(ball, VELOCIDADE_INICIAL, 0))
    heun_dist.append(tacada(ball, VELOCIDADE_INICIAL, 1))
    rk4_dist.append(tacada(ball, VELOCIDADE_INICIAL, 2))

    velocidades.append(VELOCIDADE_INICIAL)
    VELOCIDADE_INICIAL += PASSO


erro = [euler_dist[i]-heun_dist[i] for i, _ in enumerate(euler_dist)]
erro1 = [euler_dist[i]-rk4_dist[i] for i, _ in enumerate(euler_dist)]
erro2 = [heun_dist[i]-rk4_dist[i] for i, _ in enumerate(euler_dist)]
# plt.scatter(velocidades, erro, label="euler - heun")
# plt.scatter(velocidades, erro1, label="euler - rk4")
# plt.scatter(velocidades, erro2, label="rk4 - heun")
# plt.xlabel("velocidade (pixels/segundo)")
# plt.ylabel("distância (pixels)")
# plt.title("Distância entre os Métodos Euler - Heun - RK4")
# plt.legend()
# plt.show()

err_rel = [erro2[i]/euler_dist[i]*100 for i, _ in enumerate(euler_dist)]
plt.scatter(euler_dist, err_rel)
plt.xlabel("distância (pixels)")
plt.ylabel("erro relativo: (euler-rk4)/euler % ")
plt.title("distância relativa entre os métodos Euler e RK4")
plt.legend()
plt.show()

