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

pro vis, p, i, norm=norm, png=png, sz=sz

  if n_elements(i) eq 0 then name =     string(p, format='(i04)') $
  else                       name = p + string(i, format='(i04)')

  W = load(name + '.raw')
  n = size(W, /dimensions)

  if not keyword_set(norm) then norm = 0
  if not keyword_set(sz  ) then sz   = n
  if 1 eq n_elements(sz  ) then sz   = [sz, sz]

  w = transpose(congrid(real_part(fft(W, /inverse)), sz[0], sz[1]))
  if norm eq 0 then w = w / max(abs(w)) $
  else              w = norm * w
  pos = sqrt( w > 0)
  neg = sqrt(-w > 0)

  img        = fltarr(3, sz[0], sz[1])
  img[0,*,*] = 1024 * (pos   + neg^3) < 255
  img[1,*,*] = 1024 * (pos^2 + neg^2) < 255
  img[2,*,*] = 1024 * (pos^3 + neg  ) < 255
  
  if keyword_set(png) then begin
    write_png, name + '.png', img
    if !d.window ne -1 then tv, img, /true
  endif else begin
    if !d.window eq -1 then window, retain=2, xSize=sz[0], ySize=sz[1]
    tv, img, /true 
  endelse

end
