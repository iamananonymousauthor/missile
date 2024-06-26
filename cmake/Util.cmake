macro(__tvm_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} CACHE STRING ${description})
  endif()
endmacro()

#######################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   tvm_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
macro(tvm_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        __tvm_option(${variable} "${description}" ON)
      else()
        __tvm_option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        __tvm_option(${variable} "${description}" ON)
      else()
        __tvm_option(${variable} "${description}" OFF)
      endif()
    else()
      __tvm_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()

function(assign_source_group group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "." "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "." "\\" _source_path_msvc "${_source_path}")
        source_group("${group}\\${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)
