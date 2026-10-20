{
  description = "An FSM controller that integrates reinforcement learning policies with mc_rtc for robotic control. This package provides example policies for the H1 humanoid robot. It currently only supports ONNX format for policy deployment.";

  inputs = {
    mc-rtc-nix.url = "github:mc-rtc/nixpkgs";
    flake-parts.follows = "mc-rtc-nix/flake-parts";
    systems.follows = "mc-rtc-nix/systems";

    # ccache-trigger.follows = "mc-rtc-nix/ccache-trigger";
    ccache-trigger.url = "github:boolean-option/true";

    # TO MERGE
    tvm.url = "github:jrl-umi3218/TVM/pull/59/head";
    tvm.inputs.mc-rtc-nix.follows = "mc-rtc-nix";

    # Explicit Compliance and Safety

    # does not contain torque task
    # mc-rtc.url = "github:jrl-umi3218/mc_rtc/pull/531/head";

    # contains torque task but not up to date
    # mc-rtc.url = "github:bastien-muraccioli/mc_rtc";
    mc-rtc.url = "github:arntanguy/mc_rtc?ref=bastien-master"; # bastien's master rebased on top of mc_rtc@master
    mc-rtc.inputs.mc-rtc-nix.follows = "mc-rtc-nix";

    # state-observation.url = "github:jrl-umi3218/state-observation/pull/34/head";
    # FIXME: what PRs does this correspond to?
    state-observation.url = "github:bastien-muraccioli/state-observation";
    state-observation.inputs.mc-rtc-nix.follows = "mc-rtc-nix";

    #mc-state-observation.url = "github:jrl-umi3218/mc_state_observation/pull/60/head";
    # FIXME: what PRs does this correspond to?
    mc-state-observation.url = "github:bastien-muraccioli/mc_state_observation";
    mc-state-observation.flake = false;

    # replaces isri-aist/mc_joystick_plugin/pull/3 with a PR against the correct remote
    # rebased on top of isri-aist/mc_joystick_plugin@master for jrl-cmakemodulesv2
    mc-joystick-plugin.url = "github:isri-aist/mc_joystick_plugin/pull/3/head";
    mc-joystick-plugin.flake = false;

    # MERGED tasks.url = "github:jrl-umi3218/Tasks/pull/112/merge";
    tasks.url = "github:jrl-umi3218/Tasks";
    tasks.inputs.mc-rtc-nix.follows = "mc-rtc-nix";

  };

  outputs =
    inputs:
    /**
      This will:
      - Build the controller from the local sources in this repository
      - Generate two mc-rtc-superbuild development shells:
        - mc-rtc-superbuild-controller-name : a release shell allowing to run the controller
        - mc-rtc-superbuild-controller-name-devel : a development shell allowing to build the controller and run it
        Instructions are displayed upon entering the shells.

       You can control the behaviour or define your own shells with:
       mc-rtc-nix = {};
       mc-rtc-superbuild = {};

       Note that the mc-rtc-superbuild attribute set will be merged with the default controller configuration
    */
    inputs.mc-rtc-nix.lib.mkMcRtcController inputs "rl-controller" (
      { lib, ... }:
      {
        mc-rtc-nix.overlays.ccache = inputs.ccache-trigger.value;
        flakoboros = {
          overrideAttrs = {
            rl-controller = {
              src = lib.cleanSource ./.;
            };
            tvm = {
              src = inputs.tvm;
            };
            tasks = {
              src = inputs.tasks;
            };
            mc-rtc = {
              src = inputs.mc-rtc;
            };
            state-observation = {
              src = inputs.state-observation;
            };
            mc-h1 = { pkgs-final, ... }: {
              version = "1.1.0";
              src = pkgs-final.fetchFromGitHub {
                owner = "isri-aist";
                repo = "mc_h1";
                tag = "v1.1.0";
                hash = "sha256-19AAvMG2QTXRhKS+C6KiOWuABrPQ6HKx5ik8V432+xg=";
              };
            };
            h1-description = { pkgs-final, ... }: {
              version = "1.1.0";
              src = pkgs-final.fetchFromGitHub {
                owner = "isri-aist";
                repo = "h1_description";
                tag = "v1.1.0";
                hash = "sha256-wrM2rpIYcUyy0HWCOEnB73qwRR8jgkw8WMrp9irj5WA=";
              };

            };
            mc-state-observation = { pkgs-final, drv-prev, ... }: {
              src = inputs.mc-state-observation;
              buildInputs = (drv-prev.buildInputs or [ ]) ++ [ pkgs-final.jrl-cmakemodules ];
            };
            mc-joystick-plugin = {
              src = inputs.mc-joystick-plugin;
              dontWrapQtApps = true; # FIXME
            };
          };
          packages = {
            rl-controller =
              {
                stdenv,
                mkMcRtcController,
                lib,
                fetchFromGitHub,
                cmake,
                mc-rtc,
                jrl-cmakemodules,
                mc-joystick-plugin,
                onnxruntime,
              }:
              mkMcRtcController rec {
                pname = "rl-controller";
                version = "0.0.0";

                src = fetchFromGitHub {
                  owner = "bastien-muraccioli";
                  repo = "rl_controller";
                  rev = "d00b5ecea2b6b626c0409232f7262b34cd00c498";
                  hash = "sha256-67FVaNIdQD/0aUYLSuteLuO8jiElXLf210kzDg5WbfI=";
                };

                buildInputs = [ jrl-cmakemodules ];
                nativeBuildInputs = [ cmake ];
                propagatedBuildInputs = [
                  mc-rtc
                  mc-joystick-plugin
                  onnxruntime
                ];

                cmakeFlags = [
                  "-DONNXRUNTIME_ROOT=${onnxruntime}"
                  "-DINSTALL_DOCUMENTATION=OFF"
                  "-DMC_RTC_HONOR_INSTALL_PREFIX=ON"
                ];

                doCheck = true;

                passthru.mc-rtc = {
                  plugins = [ ]; # todo add external footstep planner plugins
                  observers = [
                    "mc-state-observation"
                    "mc-external-forces-observer"
                  ];
                  controller = {
                    Enabled = "RLController";
                    MainRobot = "H1";
                  };
                  robots = [
                    "mc-h1"
                  ];
                  suggests = {
                    plugins = [
                      "mc-joystick-plugin"
                    ];
                    # robots = [
                    #   "mc-hrp4"
                    #   "mc-hrp2"
                    #   "mc-hrp5-p"
                    #   "mc-hrp4cr"
                    #   # FIXME(mc-rhps1): too large for now
                    #   # "mc-rhps1"
                    # ];
                    apps = [
                      "mc-rtc-ticker"
                      "mc-rtc-magnum"
                    ];
                    # ++ lib.optional (!stdenv.hostPlatform.isDarwin) "mc-mujoco";
                  };
                  runApps =
                    # lib.optional (!stdenv.hostPlatform.isDarwin) "mc-mujoco" ++
                    lib.optionals (stdenv.hostPlatform.isDarwin) [
                      "mc-rtc-ticker"
                      "mc-rtc-magnum"
                    ];
                };

                meta = with lib; {
                  description = "An FSM controller that integrates reinforcement learning policies with mc_rtc for robotic control. This package provides example policies for the H1 humanoid robot. It currently only supports ONNX format for policy deployment.";
                  homepage = "https://github.com/jrl-umi3218/lipm_walking_controller";
                  license = licenses.bsd2;
                  platforms = platforms.all;
                };
              };
            mc-external-forces-observer =
              {
                stdenv,
                lib,
                cmake,
                fetchFromGitHub,
                mc-rtc,
              }:

              stdenv.mkDerivation {
                pname = "mc-external-forces-observer";
                version = "1.0.0";

                src = fetchFromGitHub {
                  owner = "isri-aist";
                  repo = "mc_external_forces_observer";
                  tag = "v1.0.0";
                  hash = "sha256-ENeW5ZcKAV2fE5ULE+OBqGcdqtWEnuNkeR4IJNsZ7WM=";
                };
                nativeBuildInputs = [
                  cmake
                ];
                propagatedBuildInputs = [
                  mc-rtc
                ];

                cmakeFlags = [ ];
                doCheck = true;

                meta = with lib; {
                  mainProgram = "mc-external-forces-observer";
                  description = "State observer for external forces based on torque measurements";
                  homepage = "https://github.com/isri-aist/mc_external_forces_observer";
                  license = licenses.bsd2;
                  platforms = platforms.all;
                };
              };
          };
        };
      }
    );
}
