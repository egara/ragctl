{
  description = "localrag — RAG system with PDF ingestion, pgvector, and FastFlowLM";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
  let
    system = "x86_64-linux";
  in
  {
    devShells."${system}".default =
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    pkgs.mkShell {
      packages = with pkgs; [
        python313
        python313Packages.pip
        python313Packages.setuptools
        python313Packages.virtualenv
        python313Packages.psycopg2
        pgadmin4-desktopmode
      ];

      shellHook = ''
        echo "Entering localrag development environment"

        VENV=venv
        if test ! -d $VENV; then
          virtualenv $VENV
        fi
        source ./$VENV/bin/activate

        if ! command -v ragctl >/dev/null 2>&1; then
          echo "Installing dependencies from requirements.txt and package..."
          pip install -r requirements.txt --quiet
          pip install -e . --quiet
        else
          echo "Dependencies and package already installed — skipping pip install"
        fi

        echo ""
        echo "  ┌────────────────────────────────────────────────────┐"
        echo "  │  localrag — RAG with PDFs, pgvector & FastFlowLM   │"
        echo "  │                                                    │"
        echo "  │  Commands:                                         │"
        echo "  │  ragctl ingest --dir data/pdfs/                    │"
        echo "  │  ragctl query \"Your question\"                      │"
        echo "  └────────────────────────────────────────────────────┘"
        echo ""

        export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.glib
        ]}:$LD_LIBRARY_PATH

        # Create data/pdfs directory if it doesn't exist
        mkdir -p data/pdfs
      '';
    };
  };
}
