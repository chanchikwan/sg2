pro vis, i, png=png

  if not keyword_set(png) then png = 0

  n1 = 1024 & m1 = 512
  n2 = 1024 & m2 = 512

  name = string(i, format='(i04)') + '.raw'
  print, 'loading: ' + name

  f = fltarr(n2, n1)
  x = findgen(n1) / n1
  y = findgen(n2) / n2

  openr, lun, name, /get_lun
  readu, lun, f & f = transpose(f)
  close, lun & free_lun, lun

  print, max(f)
  pos = ( f > 0)^.33
  neg = (-f > 0)^.33

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
