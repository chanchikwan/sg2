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

    spawn, 'wc -l log.txt', wc
    n = long(wc[0])

    data = dblarr(5, n)
    openr, lun, 'log.txt', /get_lun
    readf, lun, data
    close, lun & free_lun, lun
    data = transpose(data)

    E = data[*,0]
    y = atan(data[*,2], data[*,1])
    x = atan(data[*,4], data[*,3])

    window, 0, xSize=512, ySize=512
    plot, E

    window, 1, xSize=512, ySize=512
    plot, x, y, psym=3, /xStyle, /yStyle, xRange=!pi*[-1,1], yRange=!pi*[-1,1]

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
