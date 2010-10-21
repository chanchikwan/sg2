pro spec, i, png=png

  common func, n1, n2, f
  common spec, k1, k2, s

  if not keyword_set(png) then png = 0

  load, i

  window, 0, xSize=512, ySize=512, retain=2

  plot, [1,min([n1,n2])/2], [1e-14,1e+2], /nodata,$
        xTitle='Wavenumber k', yTitle='Shell-integrated energy spectrum E(k)',$
        /xLog, /yLog, /xStyle, /yStyle

  kk = k1^2 + k2^2
  k  = sqrt(kk)
  E  = abs(s)^2 / kk & E[0] = 0 ; the 2D spectrum E(kx,ky)

  oplot, k, E * k, psym=3, color=255*256LL^2

  sp = oned(k, E, 1.2)
  k  = sp.k
  E  = sp.E ; integrated spectrum E(k) = int E(kx,ky) k dphi

  oplot, k, 1e+2 * k^(-5./3), lineStyle=1, color=255
  oplot, k, 1e+2 * k^(-3   ), lineStyle=2, color=255*256LL
  oplot, k, 1e+2 * k^(-5   ), lineStyle=3, color=255*257LL
  oplot, k, E, thick=2

  if png then write_png, 's' + string(i, format='(i04)') + '.png', tvrd(/true)

end
