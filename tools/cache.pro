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

function cache, name, spec, quiet=quiet

  if n_elements(spec) ne 0 then begin ; write cache

    if not keyword_set(quiet) then print, 'writing cache: ' + name
    openw, lun, name, /get_lun

      printf, lun, n_elements(spec.k)
      printf, lun, spec.E
      printf, lun, spec.Z
      printf, lun, spec.k
      printf, lun, spec.n
      printf, lun, spec.b

    close,  lun & free_lun, lun
    return, spec

  endif else if file_test(name) then begin ; read cache

    if not keyword_set(quiet) then print, 'loading cache: ' + name
    openr, lun, name, /get_lun

      m = 0LL         & readf, lun, m
      E = fltarr(m)   & readf, lun, E
      Z = fltarr(m)   & readf, lun, Z
      k = fltarr(m)   & readf, lun, k
      n = lonarr(m)   & readf, lun, n 
      b = fltarr(m+1) & readf, lun, b

    close, lun & free_lun, lun
    return, {E:E, Z:Z, k:k, n:n, b:b}

  endif else return, []

end
