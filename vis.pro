pro vis, i, png=png

  common func, n1, n2, f

  if not keyword_set(png) then png = 0

  m1 = 512
  m2 = 512

  load, i

  print, max(f)
  pos = 4 * ( f / sqrt(n1 * n2) > 0)^.33
  neg = 4 * (-f / sqrt(n1 * n2) > 0)^.33

  img = fltarr(3,n1,n2)
  img[0,*,*] =       pos   + .1  * neg^3
  img[1,*,*] = .33 * pos^2 + .33 * neg^2
  img[2,*,*] = .1  * pos^3 +       neg

  img = congrid(256 * img, 3, m1, m2)
  img[where(img gt 255)] = 255

  if(png) then $
    write_png, 'v' + string(i, format='(i04)') + '.png', img $
  else begin
    window, 0, xSize=m1, ySize=m2
    tv, img, /true
  endelse

end
