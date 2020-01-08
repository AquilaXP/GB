#pragma once

#ifdef OML_EXPORTS
#   define OML_API __declspec(dllexport)
#else
#   define OML_API __declspec(dllimport)
#endif
