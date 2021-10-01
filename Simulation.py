import numpy as np
from matplotlib import pyplot as plt
import time
import os
import psutil
process=psutil.Process()
                                                                    #EDIT THESE VARIABLES AS YOU PLEASE

G=6.67430*10**-11                                                   #Gravitational Constant
mass1=5.97237*10**24                                                #Mass 1 - Earth is 5.97237*10**24  
mass2=7.346*10**22                                                  #Mass 2 - Moon is 7.346*10**22   
sep=3.84402*10**8                                                   #Seperation between masses - Earth-Moon is 3.84402*10**8
L2EstTay=448600000                                                  #Massless body starting distance (Taylor) - Literature value for stable body orbit is 448600000, theoretical value is orbRad[1]*(1+(mass1/(3*mass2))**(1/3))
L2EstRK=448600000                                                   #Massless body starting distance (RK) 
nOrbitsTay=2                                                        #Number of orbits simulated (Taylor) (minimum 1)
nOrbitsRK=2                                                         #Number of orbits simulated (RK) (minimum 1)
stepsTay=100000                                                     #Number of timesteps per orbit (Taylor)
stepsRK=100000                                                      #Number of timesteps per orbit (RK)


def calcT(G,mass1,mass2,sep):                                       #Returns orbital period of 2 masses                          
    return(2*np.pi*(sep**3/(G*(mass1+mass2)))**0.5)

def calcW(T):                                                       #Returns angular frequency of an orbit
    return(2*np.pi/T)

def calcOrbRad(mass1,mass2,sep):                                    #Returns orbital radius 2 masses
    return[mass2*sep/(mass1+mass2),mass1*sep/(mass1+mass2)]

def SHM(r,w,t):                                                     #Returns [x,y] co-ordiantes of a point in an orbit
    return[r*np.cos(w*t),r*np.sin(w*t)]

def calcAcc(G,mass1,mass2,p,p1,p2):                                 #Returns [x,y] acceleration of an object given positions of the object and 2 masses
    rad1Cubed=((p[0]-p1[0])**2+(p[1]-p1[1])**2)**1.5
    rad2Cubed=((p[0]-p2[0])**2+(p[1]-p2[1])**2)**1.5
    xComp1=(p[0]-p1[0])/rad1Cubed
    xComp2=(p[0]-p2[0])/rad2Cubed
    yComp1=(p[1]-p1[1])/rad1Cubed
    yComp2=(p[1]-p2[1])/rad2Cubed
    return[-G*(mass1*xComp1+mass2*xComp2),-G*(mass1*yComp1+mass2*yComp2)]
    
def posTay(p,pDot1,pDot2,dt):                                       #Uses Taylor series to return new approximate location
    return(p+dt*pDot1+dt*pDot2/2)

def simTay(L2EstTay):                                               #Using Taylor Series, simulates and plots movement of massless object about 2 bodies in circular orbit
    a1=time.time()
    b1=process.memory_info().rss
    steps=100000
    dt=T/steps
    rPos=[[L2EstTay,0]]
    rVel=[0,w*L2EstTay]
    print('rPos=',rPos[0])
    print('rVel=',rVel)
    L2Pos=[[L2EstTay,0]]
    pos1=[[-orbRad[0],0]]
    pos2=[[orbRad[1],0]]
    c=0
    while c<steps:
        acc=calcAcc(G,mass1,mass2,rPos[-1],pos1[-1],pos2[-1])
        rPos.append([posTay(rPos[-1][0],rVel[0],acc[0],dt),posTay(rPos[-1][1],rVel[1],acc[1],dt)])
        rVel=[rVel[0]+dt*acc[0],rVel[1]+dt*acc[1]]
        c+=1
        L2Pos.append(SHM(L2EstTay,w,c*dt))
        pos1.append(SHM(-orbRad[0],w,c*dt))
        pos2.append(SHM(orbRad[1],w,c*dt))
    c2=0
    while c<steps*nOrbitsTay:                                                        
        acc=calcAcc(G,mass1,mass2,rPos[-1],pos1[c2],pos2[c2])
        rPos.append([posTay(rPos[-1][0],rVel[0],acc[0],dt),posTay(rPos[-1][1],rVel[1],acc[1],dt)])
        rVel=[rVel[0]+dt*acc[0],rVel[1]+dt*acc[1]]
        c+=1
        c2=c%steps
    a2=time.time()
    b2=process.memory_info().rss
    plt.plot([i[0] for i in pos1], [i[1] for i in pos1],label='Mass 1 Orbit')
    plt.plot([i[0] for i in pos2],[i[1] for i in pos2],label='Mass 2 Orbit')
    plt.plot([i[0] for i in L2Pos],[i[1] for i in L2Pos],label='Circular Orbit')
    plt.plot([i[0] for i in rPos],[i[1] for i in rPos],label='Massless Body')
    plt.legend()
    b3=process.memory_info().rss
    print('rPos Final =',rPos[-1])
    print('rVel Final =',rVel)
    print('Tay Overshoot=',(rPos[-1][0]**2+rPos[-1][1]**2)**0.5-L2EstTay)
    print('Tay Memory = %.1f Mb'%((b2-b1)/1000000))
    print('Tay Plot Memory = %.1f Mb'%((b3-b2)/1000000))
    print('Tay Total Memory = %.1f Mb'%((b3-b1)/1000000))
    print('Currently Used Memory = %.1f Mb'%((b3)/1000000))
    print('Tay Time = %.1f s'%(a2-a1))
    plt.show()                                                      #Shows plot, HASH THIS WHILE AUTOMATING
    #plt.savefig('TayFigure.png')
    #return(rPos[-1])                                               #UNHASH THIS WHILE AUTOMATING

def posVelRK(mass1,mass2,p,pDot1,pDot2,p1,p2,dt):                   #Uses RK method to return new approximate location
    z1=[p[0]+dt*pDot1[0]/2,p[1]+dt*pDot1[1]/2]
    z1Dot1=[pDot1[0]+dt*pDot2[0]/2,pDot1[1]+dt*pDot2[1]/2]
    z1Dot2=calcAcc(G,mass1,mass2,z1,p1,p2)
    z2=[p[0]+dt*z1Dot1[0]/2,p[1]+dt*z1Dot1[1]/2]
    z2Dot1=[pDot1[0]+dt*z1Dot2[0]/2,pDot1[1]+dt*z1Dot2[1]/2]
    z2Dot2=calcAcc(G,mass1,mass2,z2,p1,p2)
    z3=[p[0]+dt*z2Dot1[0],p[1]+dt*z2Dot1[1]]
    z3Dot1=[pDot1[0]+dt*z2Dot2[0],pDot1[1]+dt*z2Dot2[1]]
    z3Dot2=calcAcc(G,mass1,mass2,z3,p1,p2)
    posX=p[0]+dt*(pDot1[0]+2*z1Dot1[0]+2*z2Dot1[0]+z3Dot1[0])/6
    posY=p[1]+dt*(pDot1[1]+2*z1Dot1[1]+2*z2Dot1[1]+z3Dot1[1])/6
    velX=pDot1[0]+dt*(pDot2[0]+2*z1Dot2[0]+2*z2Dot2[0]+z3Dot2[0])/6
    velY=pDot1[1]+dt*(pDot2[1]+2*z1Dot2[1]+2*z2Dot2[1]+z3Dot2[1])/6
    return[posX,posY,velX,velY]

def simRK(L2EstRK):                                                 #Using RK method, simulates and plots movement of massless object about 2 bodies in circular orbit
    a1=time.time()
    b1=process.memory_info().rss
    steps=100000
    dt=T/steps
    rPos=[[L2EstRK,0]]
    rVel=[0,w*L2EstRK]
    print('rPos =',rPos[0])
    print('rVel =',rVel)
    L2Pos=[[L2EstRK,0]]
    pos1=[[-orbRad[0],0]]
    pos2=[[orbRad[1],0]]
    c=0
    while c<steps:
        acc=calcAcc(G,mass1,mass2,rPos[-1],pos1[-1],pos2[-1])
        posVel=posVelRK(mass1,mass2,rPos[-1],rVel,acc,pos1[-1],pos2[-1],dt)
        rPos.append([posVel[0],posVel[1]])
        rVel=[posVel[2],posVel[3]]
        c+=1
        L2Pos.append(SHM(L2EstRK,w,c*dt))
        pos1.append(SHM(-orbRad[0],w,c*dt))
        pos2.append(SHM(orbRad[1],w,c*dt))
    c2=0
    while c<steps*nOrbitsRK:
        acc=calcAcc(G,mass1,mass2,rPos[-1],pos1[c2],pos2[c2])
        posVel=posVelRK(mass1,mass2,rPos[-1],rVel,acc,pos1[c2],pos2[c2],dt)
        rPos.append([posVel[0],posVel[1]])
        rVel=[posVel[2],posVel[3]]
        c+=1
        c2=c%steps
    a2=time.time()
    b2=process.memory_info().rss
    plt.plot([i[0] for i in pos1], [i[1] for i in pos1],label='Mass 1 Orbit')
    plt.plot([i[0] for i in pos2],[i[1] for i in pos2],label='Mass 2 Orbit')
    plt.plot([i[0] for i in L2Pos],[i[1] for i in L2Pos],label='Circular Orbit')
    plt.plot([i[0] for i in rPos],[i[1] for i in rPos],label='Massless Body')
    plt.legend()
    b3=process.memory_info().rss
    print('rPos Final =',rPos[-1])
    print('rVel Final =',rVel)
    print('RK Overshoot=',(rPos[-1][0]**2+rPos[-1][1]**2)**0.5-L2EstRK)
    print('RK Memory = %.1f Mb'%((b2-b1)/1000000))
    print('RK Plot Memory = %.1f Mb'%((b3-b2)/1000000))
    print('RK Total Memory = %.1f Mb'%((b3-b1)/1000000))
    print('Currently Used Memory = %.1f Mb'%((b3)/1000000))
    print('RK Time = %.1f s'%(a2-a1))
    plt.show()                                                      #Shows plot, HASH THIS WHILE AUTOMATING
    #plt.savefig('RKFigure.png')
    #return(rPos[-1])                                               #UNHASH THIS WHILE AUTOMATING

def autoTay(low,high):                                              #Recurses Taylor process to find optimal orbital radius (i.e. L2 Lagrange point)
    d=(low+high)/2
    c1=simTay(d)
    c2=(c1[0]**2+c1[1]**2)**0.5
    c3=c2-d
    print('Low=',low)
    print('High=',high)
    print('Mid=',d)
    print('Tay Overshoot=',c3)
    print('\n')
    if c3>0:
        high=d
    else:
        low=d
    autoTay(low,high)
#autoTay(440000000,450000000)                                       #UNHASH TO AUTOMATE

def autoRK(low,high):                                               #Recurses RK process to find optimal orbital radius (i.e. L2 Lagrange point)
    d=(low+high)/2
    c1=simRK(d)
    c2=(c1[0]**2+c1[1]**2)**0.5
    c3=c2-d
    print('Low=',low)
    print('High=',high)
    print('Mid=',d)
    print('RK Overshoot=',c3)
    print('\n')
    if c3>0:
        high=d
    else:
        low=d
    autoRK(low,high)
#autoRK(440000000,450000000)                                        #UNHASH TO AUTOMATE


T=calcT(G,mass1,mass2,sep)                                          #DO NOT EDIT
w=calcW(T)                                                          #DO NOT EDIT
orbRad=calcOrbRad(mass1,mass2,sep)                                  #DO NOT EDIT


simTay(L2EstTay)                                                    #Runs the Taylor code, HASH THIS WHILE AUTOMATING
print('\n')                                                         
plt.clf()                                                           
simRK(L2EstRK)                                                      #Runs the RK code, HASH THIS WHILE AUTOMATING
