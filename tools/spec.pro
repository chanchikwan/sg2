;; Copyright (C) 2010-2011 Chi-kwan Chan
;; Copyright (C) 2010-2011 NORDITA
;;
;; This file is part of sg2.
;;
;; Sg2 is free software: you can redistribute it and/or modify it
;; under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; Sg2 is distributed in the hope that it will be useful, but WITHOUT
;; ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
;; or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
;; License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with sg2.  If not, see <http://www.gnu.org/licenses/>.

pro spec, p, i, n=n, eps=eps, png=png, lego=lego

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')
  if not keyword_set(lego) then lego = 0 else lego = 10

  s = cache(name + '.sca')
  if n_elements(s) eq 0 then begin ; load data
    W = load(name + '.raw')
    Z = 0.5 * abs(W)^2 ; the enstrophy
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
  plot, [1,max(s.b)], [1e-12,1e+2], /nodata, /xStyle, /yStyle, /xLog, /yLog, $
        xTitle='Wavenumber k', title=name, $
        yTitle='Shell-integrated energy spectrum E(k)'
  
  oplot, s.k, 1e+2 * s.k^(-5./3), lineStyle=1
  oplot, s.k, 1e+2 * s.k^(-3   ), lineStyle=2
  oplot, s.k, 1e+2 * s.k^(-5   ), lineStyle=3

  oplot, s.k, s.E, thick=2, psym=lego

  ; clean up device
  if keyword_set(eps) then begin
    device, /close
    set_plot, 'x'
  endif else if keyword_set(png) then begin
    write_png, name + '.png', tvrd(/true)
  endif

end
