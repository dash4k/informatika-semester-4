<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>WarJa</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css" />
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
</head>
<body class="bg-gray-100 p-8">

  <div class="max-w-5xl mx-auto">
    <h1 class="text-2xl font-bold mb-4">Manajemen Pengguna</h1>
    <div class="w-full flex flex-row justify-end gap-1">
        <button onclick="refresh()" class="mb-4 px-4 py-1 bg-blue-600 hover:bg-blue-300 text-white rounded"><i class="fas fa-sync-alt"></i></button>
        <button onclick="showForm()" class="mb-4 px-4 py-1 bg-green-600 hover:bg-green-300 text-white rounded"><i class="fa-solid fa-plus"></i></button>
    </div>

    <!-- Add/Edit Form -->
    <div id="form-container" class="fixed inset-0 bg-black bg-opacity-30 backdrop-blur-sm z-50 flex items-center justify-center opacity-0 pointer-events-none transition-all duration-300 ease-out">
        <div class="bg-white p-6 rounded shadow max-w-lg w-full transform transition-all duration-300 ease-in-out">
            <form id="user-form" class="flex flex-col gap-2">
                <div class="mb-2 flex flex-row justify-center items-start gap-1">
                    <div class="w-1/2">
                        <label for="id" class="block mb-2 text-sm font-light text-gray-900">ID</label>
                        <input type="number" id="id" name="user_id" placeholder="xx08561xxx" class="border p-2 w-full rounded-md">
                        <p id="idErrorMessage" class="text-red-500 mt-1 text-xs"></p>
                    </div>
                    <div class="w-1/2">
                        <label for="role" class="block mb-2 text-sm font-light text-gray-900">Role</label>
                        <select id="role" name="role" class="border p-2 py-3 w-full rounded-md bg-white h-auto">
                        <option value="user">User</option>
                        <option value="dosen">Dosen</option>
                        <option value="admin">Admin</option>
                        </select>
                    </div>
                </div>
                <div class="mb-2 flex flex-row justify-center items-start gap-1">
                    <div class="w-1/2 grow-0">
                        <label for="email" class="block mb-2 text-sm font-light text-gray-900">Email</label>
                        <input type="email" id="email" name="email" placeholder="x@student.unud.ac.id" class="border p-2 w-full rounded-md">
                        <p id="emailErrorMessage" class="text-red-500 mt-1 text-xs"></p>
                    </div>
                    <div class="w-1/2 grow-0">
                        <label for="password" class="block mb-2 text-sm font-light text-gray-900">Password</label>
                        <input type="password" id="password" name="password" placeholder="********" class="border p-2 w-full rounded-md">
                        <p id="passwordErrorMessage" class="text-red-500 mt-1 text-xs"></p>
                    </div>
                </div>
                <div class="mb-2">
                    <label for="name" class="block mb-2 text-sm font-light text-gray-900">Name</label>
                    <input type="text" id="name" name="name" placeholder="Agus Kopling" class="border p-2 w-full rounded-md">
                    <p id="nameErrorMessage" class="text-red-500 mt-1 text-xs"></p>
                </div>
                <div class="mb-2">
                    <label for="profile" class="block mb-2 text-sm font-light text-gray-900">Profile</label>
                    <input type="file" id="profile" name="profile" class="border p-2 w-full rounded-md">
                    <p class="mt-2 text-xs text-gray-500">SVG, PNG, or JPG (MAX. 16 MB).</p>
                    <p id="profileErrorMessage" class="text-red-500 mt-1 text-xs"></p>
                </div>
                <div class="flex gap-2 justify-end">
                    <button type="button" onclick="hideForm()" class="px-4 py-2 bg-gray-300 hover:bg-gray-200 rounded">Cancel</button>
                    <!-- <button type="button" onclick="submitForm()" class="px-4 py-2 bg-green-600 hover:bg-green-300 text-white rounded">Save</button> -->
                    <button type="submit" class="px-4 py-2 bg-green-600 hover:bg-green-300 text-white rounded">Save</button>
                </div>
            </form>
        </div>
        </div>

    <!-- Users Table -->
    <table class="w-full bg-white shadow rounded">
      <thead>
        <tr class="bg-gray-200 text-left">
          <th class="p-2">User ID</th>
          <th class="p-2">Email</th>
          <th class="p-2">Password</th>
          <th class="p-2">Name</th>
          <th class="p-2">Role</th>
          <th class="p-2 text-center">Profile</th>
          <th class="p-2 text-center">Actions</th>
        </tr>
      </thead>
      <tbody id="user-table" class="text-left"></tbody>
    </table>
  </div>
  <script>
    let editingId = null;

    const idInput = document.getElementById('id');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const nameInput = document.getElementById('name');
    const profileInput = document.getElementById('profile');

    const idErrorMsg = document.getElementById('idErrorMessage');
    const emailErrorMsg = document.getElementById('emailErrorMessage');
    const passwordErrorMsg = document.getElementById('passwordErrorMessage');
    const nameErrorMessage = document.getElementById('nameErrorMessage');
    const profileErrorMessage = document.getElementById('profileErrorMessage');

    const emailRegex = /^[a-zA-Z0-9._%+-]+@student\.unud\.ac\.id$/;

    function refresh() {
      const btn = event.currentTarget;
      btn.disabled = true;
      btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
      fetchUsers();
      setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-sync-alt"></i>';
        Swal.fire({
          icon: 'success',
          title: 'Data refreshed',
          showConfirmButton: false,
          timer: 1000
        });
      }, 500);
    }

    function fetchUsers() {
      fetch('read.php')
        .then(res => res.json())
        .then(data => {
          const tbody = document.getElementById('user-table');
          tbody.innerHTML = '';

          if (data.length > 0) {
            data.forEach(user => {
              const row = document.createElement('tr');
              row.innerHTML = `
                <td class="p-2">${user.user_id}</td>
                <td class="p-2">${user.email}</td>
                <td class="p-2">********</td>
                <td class="p-2">${user.name}</td>
                <td class="p-2">${user.role}</td>
                <td class="p-2 flex justify-center">${user.profile ? `<img src="data:image/jpeg;base64,${user.profile}" class="w-12 h-12 rounded-full">` : 'No image'}</td>
                <td class="p-2 text-center">
                  <button class="edit-btn bg-yellow-300 hover:bg-yellow-200 px-4 py-1 rounded">Edit</button>
                  <button class="delete-btn bg-red-600 hover:bg-red-300 text-white px-2 py-1 ml-2 rounded">Delete</button>
                </td>
              `;
              row.querySelector('.edit-btn').addEventListener('click', () => editUser(user));
              row.querySelector('.delete-btn').addEventListener('click', () => deleteUser(user.user_id));
              tbody.appendChild(row);
            });
          } else {
            const row = document.createElement('tr');
            row.innerHTML = `<td class="p-4 text-center text-gray-500" colspan="7">No data available</td>`;
            tbody.appendChild(row);
          }
        });
    }

    function showForm() {
      const modal = document.getElementById('form-container');
      modal.classList.remove('opacity-0', 'pointer-events-none');
      modal.classList.add('opacity-100');
      document.getElementById('user-form').reset();
      editingId = null;
      idInput.disabled = false;
    }

    function hideForm() {
      const modal = document.getElementById('form-container');
      modal.classList.remove('opacity-100');
      modal.classList.add('opacity-0', 'pointer-events-none');

      // Reset form values
      document.getElementById('user-form').reset();

      // Remove validation error styles
      idInput.classList.remove('border-red-500');
      emailInput.classList.remove('border-red-500');
      passwordInput.classList.remove('border-red-500');
      nameInput.classList.remove('border-red-500');
      profileInput.classList.remove('border-red-500');

      // Clear error messages
      idErrorMsg.innerText = "";
      emailErrorMsg.innerText = "";
      passwordErrorMsg.innerText = "";
      nameErrorMessage.innerText = "";
      profileErrorMessage.innerText = "";
    }

    function editUser(user) {
      showForm();
      idInput.disabled = true;
      editingId = user.user_id;
      idInput.value = user.user_id;
      emailInput.value = user.email;
      passwordInput.value = '';
      nameInput.value = user.name;
      document.getElementById('role').value = user.role;
    }

    function submitForm() {
      const formData = new FormData();
      formData.append('user_id', idInput.value);
      formData.append('email', emailInput.value);
      formData.append('name', nameInput.value);
      formData.append('role', document.getElementById('role').value);

      const profileFile = profileInput.files[0];
      if (profileFile) {
        formData.append('profile', profileFile);
      }

      const password = passwordInput.value.trim();
      if (password) {
        formData.append('password', password);
      }

      if (editingId) {
        formData.append('edit', 'true');
      }

      fetch(editingId ? 'update.php' : 'create.php', {
        method: 'POST',
        body: formData
      })
        .then(async res => {
          const data = await res.json();
          if (!res.ok) throw new Error(data.message || 'Unknown server error');
          hideForm();
          fetchUsers();
          Swal.fire({
            icon: 'success',
            title: editingId ? 'User Updated' : 'User Created',
            showConfirmButton: false,
            timer: 1500
          });
        })
        .catch(err => {
          Swal.fire({
            icon: 'error',
            title: 'Error',
            text: err.message
          });
        });
    }

    function deleteUser(id) {
      Swal.fire({
        title: 'Are you sure?',
        text: 'This action will permanently delete the user\'s data.',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#d33',
        cancelButtonColor: '#3085d6',
        confirmButtonText: 'Yes'
      }).then((result) => {
        if (result.isConfirmed) {
          fetch('delete.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
          })
            .then(res => res.json())
            .then(() => {
              fetchUsers();
              Swal.fire('Deleted!', 'User has been deleted.', 'success');
            })
            .catch(() => {
              Swal.fire('Error!', 'Failed to delete the user.', 'error');
            });
        }
      });
    }

    function validateId() {
      const id = idInput.value;
      let errors = [];

      if (!/^[0-9]{2}08561[0-9]{3}$/.test(id)) errors.push("xx08561xxx format");
      if (id.length < 10) errors.push("10 digits");
      if (!/^\d+$/.test(id)) errors.push("numeric only");

      if (errors.length > 0) {
        idInput.classList.add('border-red-500');
        idErrorMsg.innerText = "Id must contain: " + errors.join(", ");
        return false;
      } else {
        idInput.classList.remove('border-red-500');
        idErrorMsg.innerText = "";
        return true;
      }
    }

    function validateEmailField() {
      const email = emailInput.value;
      if (!emailRegex.test(email)) {
        emailInput.classList.add('border-red-500');
        emailErrorMsg.innerText = "Email must end with @student.unud.ac.id";
        return false;
      } else {
        emailInput.classList.remove('border-red-500');
        emailErrorMsg.innerText = "";
        return true;
      }
    }

    function validatePasswordField() {
      const password = passwordInput.value;
      let errors = [];

      if (!/[a-z]/.test(password)) errors.push("1 Lower Case");
      if (!/[A-Z]/.test(password)) errors.push("1 Upper Case");
      if (!/\d/.test(password)) errors.push("1 Number");
      if (!/[^A-Za-z\d]/.test(password)) errors.push("1 Special Character");
      if (password.length < 8) errors.push("8 Characters long");

      if (errors.length > 0) {
        passwordInput.classList.add('border-red-500');
        passwordErrorMsg.innerText = "Password must contain: " + errors.join(", ");
        return false;
      } else {
        passwordInput.classList.remove('border-red-500');
        passwordErrorMsg.innerText = "";
        return true;
      }
    }

    function validateName() {
      const name = nameInput.value.trim();
      if (name.length === 0) {
        nameInput.classList.add('border-red-500');
        nameErrorMessage.innerText = 'Name cannot be empty';
        return false;
      } else {
        nameInput.classList.remove('border-red-500');
        nameErrorMessage.innerText = '';
        return true;
      }
    }

    document.getElementById('user-form').addEventListener('submit', function (event) {
      event.preventDefault();

      const isEditing = editingId !== null;

      const idValid = validateId();
      const emailValid = validateEmailField();
      const nameValid = validateName();
      const passwordProvided = passwordInput.value.trim() !== '';
      const passwordValid = !passwordProvided || validatePasswordField();

      if (!idValid || !emailValid || !nameValid || !passwordValid) {
        return;
      }

      submitForm();
    });

    idInput.addEventListener('input', validateId);
    emailInput.addEventListener('input', validateEmailField);
    passwordInput.addEventListener('input', validatePasswordField);
    nameInput.addEventListener('input', validateName);

    fetchUsers();
  </script>
</body>
</html>
