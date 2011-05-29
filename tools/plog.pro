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

pro plog, n, Z=Z

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
    plot, t, E

    window, 1, xSize=512, ySize=512
    plot, x, y, psym=3, /xStyle, /yStyle, xRange=[0,2*!pi], yRange=[0,2*!pi]
    plots, x[n-1], y[n-1], psym=8

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
