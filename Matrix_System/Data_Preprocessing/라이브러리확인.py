import importlib.util

package_name = "adam"
package_exists = importlib.util.find_spec(package_name) is not None

if package_exists:
    print(f"{package_name} is installed.")
else:
    print(f"{package_name} is not installed.")
