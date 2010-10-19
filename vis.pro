pro vis, i

  n1 = 1024
  n2 = 1024

  name = string(i, format='(i04)') + '.raw'
  print, 'loading: ' + name

  f = fltarr(n2, n1)
  x = 2 * !pi * dindgen(n1) / n1
  y = 2 * !pi * dindgen(n2) / n2

  openr, lun, name, /get_lun
  readu, lun, f & f = transpose(f)
  close, lun & free_lun, lun

  contour, f, x, y, nlevels=64, /fill, /isotropic, /xStyle, /yStyle, /zStyle
  print, min(f), max(f)

end
