def distance_sq(r1,r2,boxl):
    dsq=0.0
    for (a,b,c) in zip(r1,r2,boxl):
        dr=a-b
        dr=dr-round(dr/c)*c
        dsq+=dr*dr
    return dsq

def categorize_SPP(x,y,z):
    d_surface=4.0
    if (x<=20.022-d_surface or x>=50.055+d_surface) and (y<=14.924-d_surface or y>=44.773+d_surface):
        return 5 #mesopore
    elif (x>20.022-d_surface and x<20.022) or (x>50.055 and x<50.055+d_surface):
        if (y>14.924-d_surface and y<14.924) or (y>44.773 and y<44.773+d_surface):
            return 3 #corner
        elif y<=14.924-d_surface or y>=44.773+d_surface:
            return 4 #surface
    elif ((y>14.924-d_surface and y<14.924) or (y>44.773 and y<44.773+d_surface)) and (x<=20.022-d_surface or x>=50.055+d_surface):
        return 4 #surface

    #inside MFI
    ncell=[4,4,3]
    boxlu=[20.022,19.899,13.383]
    boxl=[80.088,79.596,40.149]
    r0sq=2.65**2
    x0u=[-0.0050,10.0125]
    z0u=[6.6986,-0.0106]
    y0u=[4.9748,14.9242]
    dysq=[]
    for iy in range(ncell[1]):
        for k in range(2):
            y0=y0u[k]+iy*boxlu[1]
            dysq.append(distance_sq([y,],[y0,],boxl[1::2]))
    dysq_min=min(dysq)

    for ix in range(ncell[0]):
        for iz in range(ncell[2]):
            for k in range(2):
                x0=x0u[k]+ix*boxlu[0]
                z0=z0u[k]+iz*boxlu[2]

                drsq=distance_sq([x,z],[x0,z0],boxl[::2])
                if drsq<=r0sq:
                    if drsq+dysq_min<=r0sq: #l_intersection:
                        #print '0: ix=',ix,', iz=',iz,', k=',k,', center=(',x0,',',z0,')',', j=',j
                        return 0 #intersection
                    else:
                        #print '1: ix=',ix,', iz=',iz,', k=',k,', center=(',x0,',',z0,')'
                        return 1 #straight
    #print '2: dysq_min=',dysq_min,'; point=(',x,',',y,',',z,')'
    return 2 #sinusoidal

 def categorize_MFI(coord):
    x0=20.022
    y0=19.899
    z0=13.383
    r0sq=2.65**2
    dy1=coord[1]-5.06
    dy1=dy1-round(dy1/y0)*y0
    dy2=coord[1]-14.84
    dy2=dy2-round(dy2/y0)*y0

    dx1=coord[0]-0.01
    dx1=dx1-round(dx1/x0)*x0
    dz1=coord[2]-6.65
    dz1=dz1-round(dz1/z0)*z0
    sxz=dx1**2+dz1**2
    if sxz<=r0sq: # in straight channel (0.01,y,6.65)
        if sxz+dy1**2<=r0sq or sxz+dy2**2<=r0sq: # in intersection (0.01,5.06 or 14.84,6.65)
            return 0 #intersection
        else:
            return 1 #straight
    else:
        dx1=coord[0]-10.02
        dx1=dx1-round(dx1/x0)*x0
        dz1=coord[2]-0.04
        dz1=dz1-round(dz1/z0)*z0
        sxz=dx1**2+dz1**2
        if sxz<=r0sq: # in straight channel (10.02,y,0.04)
            if sxz+dy1**2<=r0sq or sxz+dy2**2<=r0sq: # in intersection (10.02,5.06 or 14.84,0.04)
                return 0 #intersection
            else:
                return 1 #straight
        else:
            return 2 #sinusoidal
