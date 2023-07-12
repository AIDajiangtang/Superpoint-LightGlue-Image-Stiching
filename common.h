#ifndef _COMMON_H_
#define _COMMON_H_

#    ifdef FEATURE_MATCHER_EXPORTS
/* We are building this library */
#      define FEATURE_MATCHER_EXPORTS __declspec(dllexport)
#    else
/* We are using this library */
#      define FEATURE_MATCHER_EXPORTS __declspec(dllimport)
#    endif


#endif // !_COMMON_H_