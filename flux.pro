pro flux, p, i, eps=eps, png=png

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')

  FZ = cache(name + '.fzc')
  FE = cache(name + '.fec')
  if n_elements(FZ) eq 0 or n_elements(FE) eq 0 then begin ; load data
    d  = load(name + '.raw', /nonl)
    kk = getkk(d.W)
    k  = sqrt(kk)
    TZ = real_part(conj(d.W) * d.J) ; the enstrophy transfer
    TE = TZ / kk & TE[0] = 0.0      ; the energy transfer
    FZ = cache(name + '.fzc', oned(k, -TZ, 250, /cumul))
    FE = cache(name + '.fec', oned(k, -TE, 250, /cumul))
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
  plot, [1,max(FZ.b)], [-1.2,1.2], /nodata, /xStyle, /yStyle, /xLog, $
        xTitle='Wavenumber k', title=name, $
        yTitle=textoidl('Normalized fluxes \Pi(k) and \Pi_Z(k)')

  oplot, FZ.k, FZ.E / (max(abs(FZ.E)) + 1e-5), thick=2, color=255
  oplot, FE.k, FE.E / (max(abs(FE.E)) + 1e-5), thick=2

  ; clean up device
  if keyword_set(eps) then begin
    device, /close
    set_plot, 'x'
  endif else if keyword_set(png) then begin
    write_png, name + '.png', tvrd(/true)
  endif

end
