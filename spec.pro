pro spec, p, i, n=n, eps=eps, png=png

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')

  s = cache(name + '.sca')
  if n_elements(s) eq 0 then begin ; load data
    W = load(name + '.raw')
    Z = 0.5 * abs(W)^2 ; the enstrophy
    if not keyword_set(n) then n = 31
    s = cache(name + '.sca', oned(Z, n))
  endif

  ; setup device
  if keyword_set(eps) then begin
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
  
  oplot, s.k, 1e+2 * s.k^(-5./3), lineStyle=1
  oplot, s.k, 1e+2 * s.k^(-3   ), lineStyle=2
  oplot, s.k, 1e+2 * s.k^(-5   ), lineStyle=3

  oplot, s.k, s.E/s.k, thick=2 ; integrated spectrum E(k) = int E(kx,ky) k dphi
                               ; divided by extra k because of the log-bins
  ; clean up device
  if keyword_set(eps) then begin
    device, /close
    set_plot, 'x'
  endif else if keyword_set(png) then begin
    write_png, name + '.png', tvrd(/true)
  endif

end
