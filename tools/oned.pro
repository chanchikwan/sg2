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

function oned, Z, n

  kk = getkk(Z)
  E  = Z / kk & E[0] = 0.0
  k  = sqrt(kk)
  kG = min(size(k, /dimensions)) / 2 + 0.99 ; Galerkin cutoff wavenumber

  ; Evenly spacing in log k
  if n_elements(n) then begin
    kb = kG^(findgen(n + 1) / n)
    kc = sqrt(kb[0:n-1] * kb[1:n])
  endif $
  ; Evenly spacing in linear k
  else begin 
    n  = ceil(kG)
    kb = findgen(n + 1) + 0.5
    kc = findgen(n) + 1
  endelse
  nb = lonarr(n)
  Zc = fltarr(n)
  Ec = fltarr(n)

  for i = 0, n-1 do begin
    case i of
      0    : j = where(0.0   lt k and k lt kb[i+1], count)
      n-1  : j = where(kb[i] le k and k le kb[i+1], count)
      else : j = where(kb[i] le k and k lt kb[i+1], count)
    endcase
    if count ne 0 then begin
      Ec[i] = total(E[j])
      Zc[i] = total(Z[j])
      nb[i] = count
    endif
  endfor

  if n_elements(n) then begin
    dk = kb[1:n] - kb[0:n-1]
    Ec = Ec / dk
    Zc = Zc / dk
  endif

  return, {E:Ec, Z:Zc, k:kc, n:nb, b:kb}

end
