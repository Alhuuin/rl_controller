#pragma once

#if defined _WIN32 || defined __CYGWIN__
#  define RLController_DLLIMPORT __declspec(dllimport)
#  define RLController_DLLEXPORT __declspec(dllexport)
#  define RLController_DLLLOCAL
#else
// On Linux, for GCC >= 4, tag symbols using GCC extension.
#  if __GNUC__ >= 4
#    define RLController_DLLIMPORT __attribute__((visibility("default")))
#    define RLController_DLLEXPORT __attribute__((visibility("default")))
#    define RLController_DLLLOCAL __attribute__((visibility("hidden")))
#  else
// Otherwise (GCC < 4 or another compiler is used), export everything.
#    define RLController_DLLIMPORT
#    define RLController_DLLEXPORT
#    define RLController_DLLLOCAL
#  endif // __GNUC__ >= 4
#endif // defined _WIN32 || defined __CYGWIN__

#ifdef RLController_STATIC
// If one is using the library statically, get rid of
// extra information.
#  define RLController_DLLAPI
#  define RLController_LOCAL
#else
// Depending on whether one is building or using the
// library define DLLAPI to import or export.
#  ifdef RLController_EXPORTS
#    define RLController_DLLAPI RLController_DLLEXPORT
#  else
#    define RLController_DLLAPI RLController_DLLIMPORT
#  endif // RLController_EXPORTS
#  define RLController_LOCAL RLController_DLLLOCAL
#endif // RLController_STATIC
