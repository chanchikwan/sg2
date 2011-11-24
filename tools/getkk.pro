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

function getkk, W

  n  = size(W, /dimensions)
  n1 = n[0]
  n2 = n[1]

  kk1 = [findgen(n1-n1/2), -reverse(findgen(n1/2)+1)]^2
  kk2 = [findgen(n2-n2/2), -reverse(findgen(n2/2)+1)]^2

  return, rebin(kk1,n1,n2) + rebin(transpose(kk2),n2,n1)

end
