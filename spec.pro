pro spec, p, i, eps=eps, png=png

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')

  s = cache(name + '.spc')
  if n_elements(s) eq 0 then begin ; load data
    W  = load(name + '.raw')
    kk = getkk(W)
    Ek = 0.5 * abs(W)^2 / kk & Ek[0] = 0 ; the 2D spectrum E(kx,ky)
    s  = cache(name + '.spc', oned(sqrt(kk), Ek, 25))
  endif

  ; setup device
  tone = 255
  if keyword_set(eps) then begin
    tone = 127
    set_plot, 'ps'
    device, filename=name + '.eps', /encap
    device, /color, /decomposed, /inch, xSize=4, ySize=4
  endif else if !d.window eq -1 then begin
    window, retain=2, xSize=512, ySize=512
  endif

  ; plot
  plot, [1,max(s.b)], [1e-14,1e+2], /nodata, /xStyle, /yStyle, /xLog, /yLog, $
        xTitle='Wavenumber k', title=name, $
        yTitle='Shell-integrated energy spectrum E(k)'
  
  oplot, s.k, 1e+2 * s.k^(-5./3), lineStyle=1, color=tone
  oplot, s.k, 1e+2 * s.k^(-3   ), lineStyle=2, color=tone*256LL
  oplot, s.k, 1e+2 * s.k^(-5   ), lineStyle=3, color=tone*257LL

  oplot, s.k, s.E/s.k, thick=2 ; integrated spectrum E(k) = int E(kx,ky) k dphi
                               ; divided byextra k because of the log-bins
  ; clean up device
  if keyword_set(eps) then begin
    device, /close
    set_plot, 'x'
  endif else if keyword_set(png) then begin
    write_png, name + '.png', tvrd(/true)
  endif

end
