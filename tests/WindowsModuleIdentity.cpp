// SPDX-License-Identifier: MIT
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

// Toolhelp declarations require the Windows SDK base types first.
#include <tlhelp32.h>

namespace {
class ModuleIdentity : public testing::EmptyTestEventListener
{
    void OnTestEnd( const testing::TestInfo& ) override
    {
        const char* enabled = std::getenv( "HDT_TEST_MODULE_IDENTITY" );
        if( !enabled || std::strcmp( enabled, "1" ) )
            return;
        const HANDLE snapshot = CreateToolhelp32Snapshot( TH32CS_SNAPMODULE, GetCurrentProcessId() );
        ASSERT_NE( snapshot, INVALID_HANDLE_VALUE );
        MODULEENTRY32W module{};
        module.dwSize = sizeof( module );
        if( Module32FirstW( snapshot, &module ) )
        {
            do
            {
                const std::wstring name = module.szModule;
                if( name.find( L"amdhip" ) != std::wstring::npos || name.find( L"amd_comgr" ) != std::wstring::npos
                    || name.find( L"hip_demand_texture" ) != std::wstring::npos
                    || name.find( L"gtest" ) != std::wstring::npos
                    || name.find( L"kpack" ) != std::wstring::npos )
                    std::wcout << L"ACTUAL_LOADED_MODULE=" << module.szExePath << L'\n';
            } while( Module32NextW( snapshot, &module ) );
            EXPECT_EQ( GetLastError(), ERROR_NO_MORE_FILES );
        }
        else
            ADD_FAILURE() << "Module enumeration failed: " << GetLastError();
        EXPECT_TRUE( CloseHandle( snapshot ) );
    }
};
const bool registered = [] {
    testing::UnitTest::GetInstance()->listeners().Append( new ModuleIdentity );
    return true;
}();
}  // namespace
#endif
