from subprocess import run

def test_ffmpeg(file_path):
    # Simple conversion to check ffmpeg functionality
    try:
        run(['ffmpeg', '-i', '', 'output.wav'], check=True)
        print("FFmpeg ran successfully.")
    except CalledProcessError as e:
        print(f"FFmpeg error: {e}")
