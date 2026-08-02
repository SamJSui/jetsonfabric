foreach(required RUNTIME_BIN OPTION VALUE EXPECTED)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

set(command "${RUNTIME_BIN}" --idle "${OPTION}" "${VALUE}")
if(DEFINED EXTRA_ARGS)
  list(APPEND command ${EXTRA_ARGS})
endif()

set(ENV{JETSONFABRIC_CLUSTER_TOKEN} "")

execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(result EQUAL 0)
  message(FATAL_ERROR "runtime unexpectedly accepted ${OPTION} ${VALUE}")
endif()

string(CONCAT output "${stdout}" "${stderr}")
string(FIND "${output}" "${EXPECTED}" expected_index)
if(expected_index EQUAL -1)
  message(FATAL_ERROR
    "runtime rejected ${OPTION} ${VALUE} without expected diagnostic\n${output}"
  )
endif()
