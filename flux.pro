pro flux, p, i, n=n, eps=eps, png=png

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')

  f = cache(name + '.fca')
  if n_elements(f) eq 0 then begin ; load data
    d = load(name + '.raw', /nonl)
    T = real_part(conj(d.W) * d.J) ; the enstrophy transfer
    if not keyword_set(n) then n = 127
    f = cache(name + '.fca', oned(-T, n))
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
  plot, [1,max(f.b)], [-1.2,1.2], /nodata, /xStyle, /yStyle, /xLog, $
        xTitle='Wavenumber k', title=name, $
        yTitle=textoidl('Normalized fluxes \Pi(k) and \Pi_Z(k)')

  fE = total(f.E, /cumulative)
  fZ = total(f.Z, /cumulative)

  oplot, f.k, fE / (max(abs(fE)) + 1e-5), thick=2
  oplot, f.k, fZ / (max(abs(fZ)) + 1e-5), thick=2, color=255

  ; clean up device
  if keyword_set(eps) then begin
    device, /close
    set_plot, 'x'
  endif else if keyword_set(png) then begin
    write_png, name + '.png', tvrd(/true)
  endif

end
