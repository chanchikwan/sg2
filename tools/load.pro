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

function load, name, quiet=quiet, nonl=nonl, show=show

  if not keyword_set(quiet) then print, 'loading: ' + name
  if not keyword_set(nonl ) then nonl  = 0
  if not keyword_set(show ) then show  = 0

  openr, lun, name, /get_lun

    ; load array information
    n  = lonarr(4) & readu, lun, n
    n1 = n[1] & h2 = n[2] & n2 = 2 * h2 - 1
    if n[0] eq -8 then h =  complexarr(h2, n1) $
    else               h = dcomplexarr(h2, n1)

    ; constructe the full vorticity
    begin
      readu, lun, h
      u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
      W = [h[0:h2-1,*], conj(u)]
      if show then tvscl, (2 * alog10(abs(W))) > (-16)
    endif

    ; constructe the full non-linear term
    if nonl then begin
      readu, lun, h
      u = reverse([[h[1:*,0]], [reverse(h[1:*,1:*],2)]])
      J = [h[0:h2-1,*], conj(u)]
      if show then tvscl, (2 * alog10(abs(J))) > (-16)
    endif

  close, lun & free_lun, lun

  if nonl then return, {W:W, J:J} $
  else         return, W

end
