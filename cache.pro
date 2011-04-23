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
