{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    {
      /*
        This example assumes your system is x86_64-linux
        change as neccesary
      */
      devShells.x86_64-linux =
        let
          pkgs = nixpkgs.legacyPackages.x86_64-linux;
          lib = pkgs.lib;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [ cmake opencv libGL stdenv.cc uv glib ];
            LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.libGL pkgs.stdenv.cc.cc.lib pkgs.glib ];
            shellHook = ''
                uv venv
                uv sync --frozen
              '';
          };
        };
    };
}
