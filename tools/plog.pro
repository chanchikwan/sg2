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

pro plog, n, Z=Z, mov=mov

  if n_elements(n) eq 0 then begin

    phi = 2 * !pi * dindgen(32) / 32
    usersym, cos(phi), sin(phi)

    spawn, 'wc -l log.txt', wc
    n = long(wc[0])

    data = dblarr(6, n)
    openr, lun, 'log.txt', /get_lun
    readf, lun, data
    close, lun & free_lun, lun
    data = transpose(data)

    t = data[*,0]
    E = data[*,1]
    y = (2 * !pi - atan(data[*,3], data[*,2])) mod (2 * !pi)
    x = (2 * !pi - atan(data[*,5], data[*,4])) mod (2 * !pi)

    window, 0, xSize=512, ySize=512
    plot, t, E, xTitle='time', yTitle='energy'

    window, 1, xSize=512, ySize=512
    if keyword_set(mov) then  begin
      tail = 5000
      for i = tail, n - 1, 1000 do begin
        plot,  x[i-tail:i], y[i-tail:i], psym=3, /iso, /xStyle, /yStyle, $
               xRange=[0,2*!pi], yRange=[0,2*!pi], $
               Title='position', xTitle='x - x(0)', yTitle='y - y(0)'
        plots, x[i], y[i], psym=8
        wait, 0.04
      endfor
    endif else begin
      plot,  x, y, psym=3, /iso, /xStyle, /yStyle, $
             xRange=[0,2*!pi], yRange=[0,2*!pi], $
             Title='position', xTitle='x - x(0)', yTitle='y - y(0)'
      plots, x[n-1], y[n-1], psym=8
    endelse

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
    window, 2, xSize=512, ySize=512
    plot, x, y, /iso, Title='Unfolded position', $
          xTitle='x - x(0)', yTitle='y - y(0)'

    window, 3, xSize=512, ySize=512
    plot, t, x^2 + y^2, xTitle='time', yTitle='[x - x(0)]^2 + [y - y(0)]^2'

  endif else begin

    E = dblarr(n + 1)
    for i = 0, n do begin
      c = cache(string(i, format='(i04)') + '.sca')
      if keyword_set(Z) then E[i] = total(c.Z) $
      else                   E[i] = total(c.E)
    endfor
    plot, E

  endelse

end
