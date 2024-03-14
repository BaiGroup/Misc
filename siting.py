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
            freqIntersection+=value
        else:
            freqStraight+=value
    else:
        dx1=coord[0]-10.02
        dx1=dx1-round(dx1/x0)*x0
        dz1=coord[2]-0.04
        dz1=dz1-round(dz1/z0)*z0
        sxz=dx1**2+dz1**2
        if sxz<=r0sq: # in straight channel (10.02,y,0.04)
            if sxz+dy1**2<=r0sq or sxz+dy2**2<=r0sq: # in intersection (10.02,5.06 or 14.84,0.04)
                freqIntersection+=value
            else:
                freqStraight+=value
        else:
            freqSinusoidal+=value
