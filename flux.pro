pro flux, p, i, eps=eps, png=png

  if not keyword_set(eps) then eps = 0 ; if eps eq 1, png has no effect
  if not keyword_set(png) then png = 0

  ; load data
  d  = load(p, i, /nonl)
  kk = getkk(d.W)
  k  = sqrt(kk)

  ; setup device
  if eps then begin
    set_plot, 'ps'
    device, filename=d.name + '.eps', /encap
    device, /color, /decomposed, /inch, xSize=4, ySize=4
  endif else if !d.window eq -1 then begin
    window, retain=2, xSize=512, ySize=512
  endif

  ; plot frame
  plot, [1,min(size(d.W, /dimensions))/2], [-1.2,1.2], /nodata, $
        xTitle='Wavenumber k', $
        yTitle=textoidl('Normalized fluxes \Pi(k) and \Pi_Z(k)'), $
        /xStyle, /yStyle, /xLog

  TZ = real_part(conj(d.W) * d.J) ; the enstrophy transfer
  TE = TZ / kk & TE[0] = 0.0      ; the energy transfer

  s = oned(k, -TZ, 250, /cumul) & FZ = s.E
  s = oned(k, -TE, 250, /cumul) & FE = s.E

  oplot, s.k, FZ / (max(abs(FZ)) + 1e-5), thick=2, color=255
  oplot, s.k, FE / (max(abs(FE)) + 1e-5), thick=2

  if eps then begin
    device, /close
    set_plot, 'x'
  endif else if png then begin
    write_png, d.name + '.png', tvrd(/true)
  endif

end
