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
;; along with sg2. If not, see <http://www.gnu.org/licenses/>.

pro setup, i, n

  common io, eps, png, name
  name = n

  if eps then begin
    set_plot, 'ps'
    device, filename=name+'.eps', /encap
    device, /color, /decomposed, /inch, xSize=4, ySize=4
  endif else if !d.window ne i then begin
    window, i, retain=2, xSize=512, ySize=512
  endif

end

pro cleanup

  common io, eps, png, name

  if eps then begin
    device, /close
    set_plot, 'x'
  endif else if png then begin
    write_png, name+'.png', tvrd(/true)
  endif

end

pro plot_cache, n, setz

  E = dblarr(n + 1)
  for i = 0, n do begin
    c = cache(string(i, format='(i04)') + '.sca')
    if setz then E[i] = total(c.Z) $
    else         E[i] = total(c.E)
  endfor

  setup, 0, 'cache'
  plot, E, xTitle='Time', yTitle='Energy'
  cleanup

end

pro plot_log, name

  phi = 2 * !pi * dindgen(32) / 32
  usersym, cos(phi), sin(phi)

  spawn, 'wc -l ' + name, wc
  n = long(wc[0])

  data = dblarr(6, n)
  openr, lun, name, /get_lun
  readf, lun, data
  close, lun & free_lun, lun
  data = transpose(data)

  t = data[*,0]
  E = data[*,1]
  y = (2 * !pi - atan(data[*,3], data[*,2])) mod (2 * !pi)
  x = (2 * !pi - atan(data[*,5], data[*,4])) mod (2 * !pi)

  setup, 0, 'path'
  plot,  x, y, psym=3, /iso, /xStyle, /yStyle, $
         xRange=[0,2*!pi], yRange=[0,2*!pi], $
         Title='Position', xTitle='x', yTitle='y'
  plots, x[n-1], y[n-1], psym=8
  cleanup

  x = x - x[0]
  y = y - y[0]
  for i = 1, n-1 do begin
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    if dx gt  1.0 then x[i:*] = x[i:*] - 2 * !pi
    if dx le -1.0 then x[i:*] = x[i:*] + 2 * !pi
    if dy gt  1.0 then y[i:*] = y[i:*] - 2 * !pi
    if dy le -1.0 then y[i:*] = y[i:*] + 2 * !pi
  endfor
  setup, 1, 'upath'
  plot, x, y, /iso, Title='Unfolded shifted position', xTitle='x', yTitle='y'
  cleanup

  setup, 2, 'r2'
  plot, t, x^2 + y^2, xTitle='Time', yTitle='x^2 + y^2'
  cleanup

end

pro plot_average, names

  setup, 2, 'r2'
  r  = 0
  r2 = 0

  for j = 0, n_elements(names) - 1 do begin
    name = names[j]+'.txt'
    spawn, 'wc -l ' + name, wc
    sz = long(wc[0])
    print, 'loading: ' + name + ', ' + string(sz) + ' lines'

    data = dblarr(6, sz)
    openr, lun, name, /get_lun
    readf, lun, data
    close, lun & free_lun, lun
    data = transpose(data)

    t = data[*,0]
    y = (2 * !pi - atan(data[*,3], data[*,2])) mod (2 * !pi)
    x = (2 * !pi - atan(data[*,5], data[*,4])) mod (2 * !pi)

    x = x - x[0]
    y = y - y[0]
    for i = 1, n_elements(t)-1 do begin
      dx = x[i] - x[i-1]
      dy = y[i] - y[i-1]
      if dx gt  1.0 then x[i:*] = x[i:*] - 2 * !pi
      if dx le -1.0 then x[i:*] = x[i:*] + 2 * !pi
      if dy gt  1.0 then y[i:*] = y[i:*] - 2 * !pi
      if dy le -1.0 then y[i:*] = y[i:*] + 2 * !pi
    endfor

    if j eq 0 then plot, [1,1e4], [0.1, 1e5], /nodata, /xLog, /yLog, $
                           xTitle='Time', yTitle=textoidl('x^2 + y^2')
    oplot, t, x^2 + y^2, color=192 * (256LL^2 + 256 + 1)
    r2 = r2 + x^2 + y^2
    r  = r  + sqrt(x^2 + y^2)
  endfor

  oplot, t, r2 / n_elements(names), thick=3, color=255
  oplot, t, (r / n_elements(names))^2, thick=3, color=255, lineStyle=2
  oplot, [1,1e4], [10,1e5], lineStyle=2

  cleanup

end

pro plog, n, Z=Z, eps=eps, png=png

  common io, seteps, setpng
  seteps = keyword_set(eps)
  setpng = keyword_set(png)

       if n_elements(n) eq 0 then plot_log, 'log.txt'         $
  else if size(n,/type) ne 7 then plot_cache, n, keyword_set(Z) $
  else                            plot_average, n

end
