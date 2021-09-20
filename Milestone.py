import numpy as np
from matplotlib import pyplot as plt
import time
import os
import psutil
process=psutil.Process()

def calcT(G,mass1,mass2,sep):                                   #Returns orbital period of 2 masses                          
    return(2*np.pi*(sep**3/(G*(mass1+mass2)))**0.5)

def calcW(T):                                                   #Returns angular frequency of an orbit
    return(2*np.pi/T)

def orbRad(mass1,mass2,sep):                                    #Returns orbital radius 2 masses
    return[mass2*sep/(mass1+mass2),mass1*sep/(mass1+mass2)]

def SHM(r,w,t):                                                 #Returns [x,y] co-ordiantes of a point in an orbit
    return[r*np.cos(w*t),r*np.sin(w*t)]

def calcAcc(G,mass1,mass2,p,p1,p2):                             #Returns [x,y] acceleration of an object given positions of the object and 2 masses
    rad1Cubed=((p[0]-p1[0])**2+(p[1]-p1[1])**2)**1.5
    rad2Cubed=((p[0]-p2[0])**2+(p[1]-p2[1])**2)**1.5
    xComp1=(p[0]-p1[0])/rad1Cubed
    xComp2=(p[0]-p2[0])/rad2Cubed
    yComp1=(p[1]-p1[1])/rad1Cubed
    yComp2=(p[1]-p2[1])/rad2Cubed
    return[-G*(mass1*xComp1+mass2*xComp2),-G*(mass1*yComp1+mass2*yComp2)]
    
def posTay(p,pDot1,pDot2,dt):                                   #Uses Taylor series to return new approximate location
    return(p+dt*pDot1+dt*pDot2/2)

def simTay(emL2EstTay):                                         #Simulates and plots orbit of massless object about 2 co-orbiting masses using Taylor series
    a1=time.time()
    b1=process.memory_info().rss
    steps=100000
    dt=emT/steps
    rPos=[[emL2EstTay,0]]
    rVel=[0,emW*emL2EstTay]
    print('rPos=',rPos[0])
    print('rVel=',rVel)
    emL2Pos=[[emL2EstTay,0]]
    ePos=[[-emOrbRad[0],0]]
    mPos=[[emOrbRad[1],0]]
    c=0
    while c<steps:
        acc=calcAcc(G,eMass,mMass,rPos[-1],ePos[-1],mPos[-1])
        rPos.append([posTay(rPos[-1][0],rVel[0],acc[0],dt),posTay(rPos[-1][1],rVel[1],acc[1],dt)])
        rVel=[rVel[0]+dt*acc[0],rVel[1]+dt*acc[1]]
        c+=1
        emL2Pos.append(SHM(emL2EstTay,emW,c*dt))
        ePos.append(SHM(-emOrbRad[0],emW,c*dt))
        mPos.append(SHM(emOrbRad[1],emW,c*dt))
    c2=0
    while c<steps*3:
        acc=calcAcc(G,eMass,mMass,rPos[-1],ePos[c2],mPos[c2])
        rPos.append([posTay(rPos[-1][0],rVel[0],acc[0],dt),posTay(rPos[-1][1],rVel[1],acc[1],dt)])
        rVel=[rVel[0]+dt*acc[0],rVel[1]+dt*acc[1]]
        c+=1
        c2=c%steps
    a2=time.time()
    b2=process.memory_info().rss
    plt.plot([i[0] for i in ePos], [i[1] for i in ePos])
    plt.plot([i[0] for i in mPos],[i[1] for i in mPos])
    plt.plot([i[0] for i in emL2Pos],[i[1] for i in emL2Pos])
    plt.plot([i[0] for i in rPos],[i[1] for i in rPos])
    b3=process.memory_info().rss
    print('rPos Final =',rPos[-1])
    print('rVel Final =',rVel)
    print('Tay Overshoot=',(rPos[-1][0]**2+rPos[-1][1]**2)**0.5-emL2EstTay)
    print('Tay Memory = %.1f Mb'%((b2-b1)/1000000))
    print('Tay Plot Memory = %.1f Mb'%((b3-b2)/1000000))
    print('Tay Total Memory = %.1f Mb'%((b3-b1)/1000000))
    print('Currently Used Memory = %.1f Mb'%((b3)/1000000))
    print('Tay Time = %.1f s'%(a2-a1))
    plt.show()
    #plt.savefig('TayFigure.png')
    #return(rPos[-1])

def posVelRK(mass1,mass2,p,pDot1,pDot2,p1,p2,dt):               #Uses RK method to return new approximate location
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

def simRK(emL2EstRK):                                           #Simulates and plots orbit of massless object about 2 co-orbiting masses using RK method
    a1=time.time()
    b1=process.memory_info().rss
    steps=100000
    dt=emT/steps
    rPos=[[emL2EstRK,0]]
    rVel=[0,emW*emL2EstRK]
    print('rPos =',rPos[0])
    print('rVel =',rVel)
    emL2Pos=[[emL2EstRK,0]]
    ePos=[[-emOrbRad[0],0]]
    mPos=[[emOrbRad[1],0]]
    c=0
    while c<steps:
        acc=calcAcc(G,eMass,mMass,rPos[-1],ePos[-1],mPos[-1])
        posVel=posVelRK(eMass,mMass,rPos[-1],rVel,acc,ePos[-1],mPos[-1],dt)
        rPos.append([posVel[0],posVel[1]])
        rVel=[posVel[2],posVel[3]]
        c+=1
        emL2Pos.append(SHM(emL2EstRK,emW,c*dt))
        ePos.append(SHM(-emOrbRad[0],emW,c*dt))
        mPos.append(SHM(emOrbRad[1],emW,c*dt))
    c2=0
    while c<steps*3:
        acc=calcAcc(G,eMass,mMass,rPos[-1],ePos[c2],mPos[c2])
        posVel=posVelRK(eMass,mMass,rPos[-1],rVel,acc,ePos[c2],mPos[c2],dt)
        rPos.append([posVel[0],posVel[1]])
        rVel=[posVel[2],posVel[3]]
        c+=1
        c2=c%steps
    a2=time.time()
    b2=process.memory_info().rss
    plt.plot([i[0] for i in ePos], [i[1] for i in ePos])
    plt.plot([i[0] for i in mPos],[i[1] for i in mPos])
    plt.plot([i[0] for i in emL2Pos],[i[1] for i in emL2Pos])
    plt.plot([i[0] for i in rPos],[i[1] for i in rPos])
    b3=process.memory_info().rss
    print('rPos Final =',rPos[-1])
    print('rVel Final =',rVel)
    print('RK Overshoot=',(rPos[-1][0]**2+rPos[-1][1]**2)**0.5-emL2EstRK)
    print('RK Memory = %.1f Mb'%((b2-b1)/1000000))
    print('RK Plot Memory = %.1f Mb'%((b3-b2)/1000000))
    print('RK Total Memory = %.1f Mb'%((b3-b1)/1000000))
    print('Currently Used Memory = %.1f Mb'%((b3)/1000000))
    print('RK Time = %.1f s'%(a2-a1))
    plt.show()
    #plt.savefig('RKFigure.png')
    #return(rPos[-1])

   
eMass=5.97237*10**24
mMass=7.346*10**22
emSep=3.84402*10**8
G=6.67430*10**-11
emT=calcT(G,eMass,mMass,emSep)
emW=calcW(emT)
emOrbRad=orbRad(eMass,mMass,emSep)

#emL2EstTay=emOrbRad[1]*(1+(mMass/(3*eMass))**(1/3))        #Mathematical Approximation
emL2EstTay=448600000                                        #Theoretical Value
simTay(emL2EstTay)
print('\n')
plt.clf()

#emL2EstRK=emOrbRad[1]*(1+(mMass/(3*eMass))**(1/3))         #Mathematical Approximation
emL2EstRK=448600000                                         #Theoretical Value
simRK(emL2EstRK)

def autoTay(low,high):                                      #Automates Taylor process to find optimal radius (L2 Lagrange point)
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
#autoTay(444246000,444247000)

def autoRK(low,high):                                       #Automates RK process to find optimal radius (L2 Lagrange point)
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
#autoRK(444246000,444247000)
