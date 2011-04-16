pro spec, pre, num, png=png, eps=eps

  if n_elements(num) eq 0 then begin
    p = ''
    i = pre
  endif else begin
    p = pre
    i = num
  endelse
  if not keyword_set(eps) then eps = 0
  if not keyword_set(png) then png = 0 ; if eps eq 1, png has no effect

  ; load data
  s  = load(p, i, /odd)
  n  = size(s, /dimensions)
  n1 = n[0]
  n2 = n[1]

  ; construct k-grid
  k1 = [dindgen(n1-n1/2), -reverse(dindgen(n1/2)+1)]
  k2 = [dindgen(n2-n2/2), -reverse(dindgen(n2/2)+1)]
  k1 =           rebin(k1, n1, n2)
  k2 = transpose(rebin(k2, n2, n1))

  ; setup
  tone = 255
  if eps then begin
    tone = 191
    set_plot, 'ps'
    device, filename=p + string(i, format='(i04)') + '.eps', /encap
    device, /color, /decomposed, /inch, xSize=4, ySize=4
  endif else $
    window, 0, xSize=512, ySize=512, retain=2

  plot, [1,min([n1,n2])/2], [1e-14,1e+2], /nodata,$
        xTitle='Wavenumber k', yTitle='Shell-integrated energy spectrum E(k)',$
        /xLog, /yLog, /xStyle, /yStyle

  kk = k1^2 + k2^2
  k  = sqrt(kk)
  E  = abs(s)^2 / kk & E[0] = 0 ; the 2D spectrum E(kx,ky)

  if not eps then oplot, k, E * k, psym=3, color=tone*256LL^2

  sp = oned(k, E, 1.2)
  k  = sp.k
  E  = sp.E ; integrated spectrum E(k) = int E(kx,ky) k dphi

  oplot, k, 1e+2 * k^(-5./3), lineStyle=1, color=tone
  oplot, k, 1e+2 * k^(-3   ), lineStyle=2, color=tone*256LL
  oplot, k, 1e+2 * k^(-5   ), lineStyle=3, color=tone*257LL
  oplot, k, E, thick=2

  if eps then begin
    device, /close
    set_plot, 'x'
  endif else if png then $
    write_png, p + string(i, format='(i04)') + '.png', tvrd(/true)

end
