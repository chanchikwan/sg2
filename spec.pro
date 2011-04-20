pro spec, p, i, eps=eps, png=png

  if not keyword_set(eps) then eps = 0 ; if eps eq 1, png has no effect
  if not keyword_set(png) then png = 0

  ; load data
  d  = load(p, i)
  kk = getkk(d.W)
  k  = sqrt(kk)

  ; setup device
  tone = 255
  if eps then begin
    tone = 191
    set_plot, 'ps'
    device, filename=d.name + '.eps', /encap
    device, /color, /decomposed, /inch, xSize=4, ySize=4
  endif else $
    window, 0, xSize=512, ySize=512, retain=2

  ; plot frame
  plot, [1,min(size(d.W, /dimensions))/2], [1e-14,1e+2], /nodata, $
        xTitle='Wavenumber k', $
        yTitle='Shell-integrated energy spectrum E(k)',$
        /xStyle, /yStyle, /xLog, /yLog

  Ek = 0.5 * abs(d.W)^2 / kk & Ek[0] = 0 ; the 2D spectrum E(kx,ky)
  if not eps then oplot, k, Ek * k, psym=3, color=tone*256LL^2

  s = oned(k, Ek, 25)
  E = s.E / s.k ; integrated spectrum E(k) = int E(kx,ky) k dphi
                ; divided by extra k because of the log-bin

  oplot, s.k, 1e+2 * s.k^(-5./3), lineStyle=1, color=tone
  oplot, s.k, 1e+2 * s.k^(-3   ), lineStyle=2, color=tone*256LL
  oplot, s.k, 1e+2 * s.k^(-5   ), lineStyle=3, color=tone*257LL
  oplot, s.k, E, thick=2

  if eps then begin
    device, /close
    set_plot, 'x'
  endif else if png then $
    write_png, d.name + '.png', tvrd(/true)

end
